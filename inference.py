import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
from utils.helper_funcs import accuracy, count_parameters, mAP, measure_inference_time
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default=None, type=Path)
    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    f_res = args.f_res
    add_noise = args.add_noise
    with (args.f_res / Path("args.yml")).open() as f:
        args = yaml.load(f, Loader=yaml.Loader)
    try:
        args = vars(args)
    except:
        if 'net' in args.keys():
            del args['net']
        args_orig = args
        args = {}
        for k, v in args_orig.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    args[kk] = vv
            else:
                args[k] = v
    args['f_res'] = f_res
    args['add_noise'] = add_noise
    with open(args['f_res'] / "args.yml", "w") as f:
        yaml.dump(args, f)
    print(args)
    #######################
    # Load PyTorch Models #
    #######################
    from modules.soundnet import SoundNetRaw as SoundNet
    ds_fac = np.prod(np.array(args['ds_factors'])) * 4
    net = SoundNet(nf=args['nf'],
                   dim_feedforward=args['dim_feedforward'],
                   clip_length=args['seq_len'] // ds_fac,
                   embed_dim=args['emb_dim'],
                   n_layers=args['n_layers'],
                   nhead=args['n_head'],
                   n_classes=args['n_classes'],
                   factors=args['ds_factors'],
                   )

    print('***********************************************')
    print("#params: {}M".format(count_parameters(net)/1e6))
    if torch.cuda.is_available() and device == torch.device("cuda"):
        t_b1 = measure_inference_time(net, torch.randn(1, 1, args['seq_len']))[0]
        print('inference time batch=1: {:.2f}[ms]'.format(t_b1))
        # t_b32 = measure_inference_time(net, torch.randn(32, 1, args['seq_len']))[0]
        # print('inference time batch=32: {:.2f}[ms]'.format(t_b32))
        print('***********************************************')

    if (f_res / Path("chkpnt.pt")).is_file():
        chkpnt = torch.load(f_res / "chkpnt.pt", map_location=torch.device(device))
        model = chkpnt['model_dict']
    else:
        raise ValueError

    if 'use_dp' in args.keys() and args['use_dp']:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.items():
            name = k.replace('module.', '')
            state_dict[name] = v
        net.load_state_dict(state_dict, strict=True)
    else:
        net.load_state_dict(model, strict=True)
    net.to(device)
    if torch.cuda.device_count() > 1:
        from utils.helper_funcs import parse_gpu_ids
        args['gpu_ids'] = [i for i in range(torch.cuda.device_count())]
        net = torch.nn.DataParallel(net, device_ids=args['gpu_ids'])
        net.to('cuda:0')
    net.eval()
    #######################
    # Create data loaders #
    #######################
    if args['dataset'] == 'esc50':
        from datasets.esc_dataset import ESCDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                fold_id=args['fold_id'],
                                transforms=None)

    elif args['dataset'] == 'speechcommands':
        from datasets.speechcommand_dataset import SpeechCommandsDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                transforms=None)

    elif args['dataset'] == 'urban8k':
        from datasets.urban8K_dataset import Urban8KDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                transforms=None,
                                fold_id=args['fold_id'])

    elif args['dataset'] == 'audioset':
        from datasets.audioset_dataset import AudioSetDataset as SoundDataset
        data_set = SoundDataset(
            args['data_path'],
            'test',
            data_subtype=None,
            segment_length=args['seq_len'],
            sampling_rate=args['sampling_rate'],
            transforms=None
        )

    else:
        raise ValueError

    if args['dataset'] != 'audioset':
        inference_single_label(net=net, data_set=data_set, args=args)
    elif args['dataset'] == 'audioset':
        inference_multi_label(net=net, data_set=data_set, args=args)
    else:
        raise ValueError("check args dataset")

def inference_single_label(net, data_set, args):
    data_loader = DataLoader(data_set,
                             batch_size=128,
                             num_workers=8,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False)

    labels = torch.zeros(len(data_loader.dataset), dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    # confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    idx_start = 0
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            _, y_est = torch.max(pred, 1)
            idx_end = idx_start + y.shape[0]
            preds[idx_start:idx_end, :] = pred
            labels[idx_start:idx_end] = y
            for t, p in zip(y.view(-1), y_est.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            print("{}/{}".format(i, len(data_loader)))
        idx_start = idx_end
    acc_av = accuracy(preds.detach(), labels.detach(), [1, ])[0]

    res = {
        "acc": acc_av,
        "preds": preds,
        "labels": labels.view(-1),
        "confusion_matrix": confusion_matrix
    }
    torch.save(res, args['f_res'] / "res.pt")

    print("acc:{}".format(np.round(acc_av*100)/100))
    print("cm:{}".format(confusion_matrix.diag().sum() / len(data_loader.dataset)))
    print('***************************************')
    bad_labels = []
    for i, c in enumerate(confusion_matrix):
        i_est = c.argmax(-1)
        if i != i_est:
            print('{} {} {}-->{}'.format(i, i_est.item(), data_set.labels[i], data_set.labels[i_est]))
            bad_labels.append([i, i_est])
    print(bad_labels)


def inference_multi_label(net, data_set, args):
    from utils.helper_funcs import collate_fn
    data_loader = DataLoader(data_set,
                             batch_size=128,
                             num_workers=8,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False,
                             collate_fn=collate_fn)

    labels = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    idx_start = 0
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to('cuda:0')
            y = [F.one_hot(torch.Tensor(y_i).long(), args['n_classes']).sum(dim=0).float() for y_i in y]
            y = torch.stack(y, dim=0).contiguous().to('cuda:0')
            pred = net(x)
            idx_end = idx_start + y.shape[0]
            preds[idx_start:idx_end, :] = torch.sigmoid(pred)
            labels[idx_start:idx_end, :] = y
            print("{}/{}".format(i, len(data_loader)))
        idx_start = idx_end
    mAP_av = mAP(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
    res = {
        "mAP": mAP_av,
        "preds": preds,
        "labels": labels.view(-1),
        "confusion_matrix": confusion_matrix
    }
    torch.save(res, args['f_res'] / "res.pt")
    # torch.save(net.state_dict(), "net.pt")
    print("mAP:{}".format(np.round(mAP_av*100)/100))


if __name__ == '__main__':
    pass