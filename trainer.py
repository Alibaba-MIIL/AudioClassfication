import yaml
import numpy as np
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.helper_funcs import accuracy, mAP
from datasets.batch_augs import BatchAugs


def parse_args():
    parser = argparse.ArgumentParser()
    '''train'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--run_name", default=None, type=Path)
    parser.add_argument('--loss_type', default="label_smooth", type=str)
    parser.add_argument('--n_epochs', default=None, type=int)
    parser.add_argument('--epoch_mix', default=None, type=int)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument("--filter_bias_and_bn", action='store_true', default=True)
    parser.add_argument("--ext_pretrained", default=None, type=str)
    parser.add_argument("--multilabel", action='store_true')
    parser.add_argument('--save_path', default=None, type=Path)
    parser.add_argument('--load_path', default=None, type=Path)
    parser.add_argument('--scheduler', default=None, type=str)
    parser.add_argument('--augs_signal', nargs='+', type=str,
                        default=['amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift'])
    parser.add_argument('--augs_noise', nargs='+', type=str,
                        default=['awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine'])
    parser.add_argument('--augs_mix', nargs='+', type=str, default=['mixup', 'timemix', 'freqmix', 'phmix'])
    parser.add_argument('--mix_loss', default='bce', type=str)
    parser.add_argument('--mix_ratio', default=0.5, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument("--kd_model", default=None, type=Path)
    parser.add_argument("--use_bg", action='store_true', default=False)
    parser.add_argument("--resume_training", action='store_true', default=False)
    parser.add_argument("--use_balanced_sampler", action='store_true', default=False)

    '''common'''
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--gpu_ids', nargs='+', default=[0])
    parser.add_argument("--use_ddp", action='store_true')
    parser.add_argument("--use_dp", action='store_true')
    parser.add_argument('--save_interval', default=100, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    '''data'''
    parser.add_argument('--fold_id', default=None, type=int)
    parser.add_argument("--data_subtype", default='balanced', type=str)
    parser.add_argument('--seq_len', default=None, type=int)
    parser.add_argument('--dataset', default=None, type=str)
    '''net'''
    parser.add_argument('--ds_factors', nargs='+', type=int, default=[4, 4, 4, 4])
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--n_layers', default=4, type=int)
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--model_type", default='SoundNetRaw', type=str)
    parser.add_argument("--nf", default=16, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)

    args = parser.parse_args()
    return args


def dummy_run(net, batch_sz, seq_len):
    print("***********Dummy Run************")
    d = next(net.parameters()).device
    x = torch.randn(batch_sz, 1, seq_len, device=d, requires_grad=False)
    t_batch = time.time()
    with torch.no_grad():
        for k in range(10):
            _ = net(x)
    t_batch = (time.time()-t_batch)/10
    print("dummy succededd, avg_time_batch:{}ms".format(t_batch*1000))
    del x
    return True


def check_args(args):
    if args.augs_noise[0] == 'none':
        args.augs_noise = []
    if args.augs_mix[0] == 'none':
        args.augs_mix = []
    return args


def create_dataset(args):
    ##################################################################################
    # ESC-50
    ##################################################################################
    if args.dataset == 'esc50':
        from datasets.esc_dataset import ESCDataset as SoundDataset
        train_set = SoundDataset(
            args.data_path,
            mode='train',
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=args.augs_signal + args.augs_noise,
            fold_id=args.fold_id
        )

        test_set = SoundDataset(
            args.data_path,
            mode='test',
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=None,
            fold_id=args.fold_id
        )

    ##################################################################################
    # SpeechCommands V2-35
    ##################################################################################
    elif args.dataset == 'speechcommands':
        from datasets.speechcommand_dataset import SpeechCommandsDataset as SoundDataset
        train_set = SoundDataset(
            args.data_path,
            mode='train',
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=args.augs_signal + args.augs_noise,
            use_background=args.use_bg
        )

        test_set = SoundDataset(
            args.data_path,
            mode='val',
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=None,
            use_background=False
        )

    ##################################################################################
    # AudioSet
    ##################################################################################
    elif args.dataset == 'audioset':
        from datasets.audioset_dataset import AudioSetDataset as SoundDataset

        train_set = SoundDataset(
            args.data_path,
            'train',
            data_subtype=args.data_subtype,
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=args.augs_signal + args.augs_noise,
        )

        test_set = SoundDataset(
            args.data_path,
            'test',
            data_subtype=None,
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=None,
        )

    ##################################################################################
    # Urban8K
    ##################################################################################
    elif args.dataset == 'urban8k':
        from datasets.urban8K_dataset import Urban8KDataset as SoundDataset

        train_set = SoundDataset(
            args.data_path,
            'train',
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=args.augs_signal + args.augs_noise,
            fold_id=args.fold_id,
        )

        test_set = SoundDataset(
            args.data_path,
            'test',
            segment_length=args.seq_len,
            sampling_rate=args.sampling_rate,
            transforms=None,
            fold_id=args.fold_id
        )

    return train_set, test_set


def create_model(args):
    from modules.soundnet import SoundNetRaw as SoundNet
    ds_fac = np.prod(np.array(args.ds_factors)) * 4
    net = SoundNet(nf=args.nf,
                   dim_feedforward=args.dim_feedforward,
                   clip_length=args.seq_len // ds_fac,
                   embed_dim=args.emb_dim,
                   n_layers=args.n_layers,
                   nhead=args.n_head,
                   n_classes=args.n_classes,
                   factors=args.ds_factors,
                   )
    return net


def save_model(net, opt, loss, best_loss, acc, best_acc, steps, root, lr_scheduler=None, scaler=None):
    if acc > best_acc:
        best_acc = acc
        best_loss = loss
        chkpnt = {
            'best_acc': best_acc,
            'model_dict': net.state_dict(),
            'opt_dict': opt.state_dict(),
            'steps': steps,
            'best_loss': best_loss,
        }
        if lr_scheduler is not None:
            chkpnt['lr_scheduler'] = lr_scheduler.state_dict()
        if scaler is not None:
            chkpnt['scaler'] = scaler.state_dict()
        torch.save(chkpnt, root / "chkpnt.pt")
        torch.save(net.state_dict(), root / "best_model.pt")
        print(best_acc, 'saved')

    elif acc == best_acc:
        if loss < best_loss:
            best_loss = loss
            chkpnt = {
                'best_acc': best_acc,
                'model_dict': net.state_dict(),
                'opt_dict': opt.state_dict(),
                'steps': steps,
                'best_loss': best_loss,
            }
            if lr_scheduler is not None:
                chkpnt['lr_scheduler'] = lr_scheduler.state_dict()
            torch.save(chkpnt, root / "chkpnt.pt")
            torch.save(net.state_dict(), root / "best_model.pt")
            print(best_acc, 'saved')
    return best_acc, best_loss


def train(args):
    if args.dataset == 'esc50':
        args.data_path = r'../data/ESC/ESC-50'
        args.sampling_rate = 22050
        args.n_classes = 50
    elif args.dataset == 'audioset':
        args.data_path = r'../data/audioset'
        args.sampling_rate = 22050
        args.n_classes = 527
    elif args.dataset == 'speechcommands':
        args.data_path = r'../data/SpeechCommands/speech_commands_v0.02'
        args.sampling_rate = 16000
        args.n_classes = 35
    elif args.dataset == 'urban8k':
        args.data_path = r'../data/UrbanSound8K'
        args.sampling_rate = 22050
        args.n_classes = 10
    else:
        raise ValueError("Wrong dataset in data")

    #######################
    # Create data loaders #
    #######################
    train_set, test_set = create_dataset(args)

    if args.multilabel:
        from utils.helper_funcs import collate_fn
        if args.use_balanced_sampler:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set.samples_weight, train_set.__len__(), replacement=True)
            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=False,
                                      drop_last=True,
                                      collate_fn=collate_fn,
                                      sampler=sampler
                                      )
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=collate_fn,
                                      )
        test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 )
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  shuffle=False if train_set is None else True,
                                  drop_last=True,
                                  )
        test_loader = DataLoader(test_set,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 )

    #####################
    # Network           #
    #####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ba_params = {
        'seq_len': args.seq_len,
        'fs': args.sampling_rate,
        'device': device,
        'augs': args.augs_mix,
        'mix_ratio': args.mix_ratio,
        'batch_sz': args.local_rank,
        'epoch_mix': args.epoch_mix,
        'resample_factors': [0.8, 0.9, 1.1, 1.2],
        'multilabel': True if args.multilabel else False,
        'mix_loss': args.mix_loss
    }
    batch_augs = BatchAugs(ba_params)

    if args.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(init_scale=2**10)
        eps = 1e-4
    else:
        scaler = None
        eps = 1e-8

    #####################
    # Network           #
    #####################
    net = create_model(args)
    net.to(device)
    ####################################
    # ext pretrainining           #
    #############pretrained#######################
    if args.ext_pretrained is not None:
        pre = ''
        print("loading model for pretraining ", (Path(pre + args.ext_pretrained) / Path("model.pt")).is_file())
        net_ext = torch.load(Path(pre + args.ext_pretrained) / Path("model.pt"))
        with (args.ext_pretrained / Path("args.yml")).open() as f:
            args_pretrained = yaml.load(f, Loader=yaml.Loader)
        try:
            args_pretrained = vars(args_pretrained)
        except:
            pass
        from modules.soundnet import SoundNetRaw as SoundNet
        ds_fac = np.prod(np.array(args_pretrained['ds_factors'])) * 4
        net = SoundNet(
            nf=args['nf'],
            dim_feedforward=args['dim_feedforward'],
            clip_length=args['seq_len'] // ds_fac,
            embed_dim=args['emb_dim'],
            n_layers=args['n_layers'],
            nhead=args['n_head'],
            n_classes=args['n_classes'],
            factors=args['ds_factors'],
             )
        try:
            net.load_state_dict(net_ext, strict=True)
        except:
            '''remove module. prefix in case of DataParallel module'''
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k, v in net_ext.items():
                name = k.replace('module.', '')
                state_dict[name] = v
            else:
                net.load_state_dict(state_dict, strict=True)
        del net_ext
        nn = args.seq_len // (np.prod(np.array(args.ds_factors)) * 4) + 1
        net.tf.pos_embed.data = F.interpolate(net.tf.pos_embed.data.transpose(2, 1), size=nn).transpose(2, 1)
        net.tf.fc = torch.nn.Linear(args.emb_dim, args.n_classes)
        net.to(device)

    if args.kd_model:
        print("Loading teacher model {}".format(args.kd_model))
        with (args.kd_model / Path("args.yml")).open() as f:
            args_t = yaml.load(f, Loader=yaml.Loader)
        try:
            args_t = vars(args_t)
        except:
            pass
        from modules.soundnet import SoundNetRaw as SoundNet
        net_t = SoundNet(
            nf=args_t['nf'],
            dim_feedforward=args_t['dim_feedforward'],
            clip_length=args_t['seq_len'] // ds_fac,
            embed_dim=args_t['emb_dim'],
            n_layers=args_t['n_layers'],
            nhead=args_t['n_head'],
            n_classes=args_t['n_classes'],
            factors=args_t['ds_factors']
        )
        if (args.kd_model / Path('model.pt')).is_file():
            teacher = torch.load(args.kd_model / Path('model.pt'), map_location=torch.device(device))
        else:
            chkpnt = torch.load(args.kd_model / Path('chkpnt.pt'), map_location=torch.device(device))
            teacher = chkpnt['model_dict']
        try:
            net_t.load_state_dict(teacher, strict=True)
        except:
            '''remove module. prefix in case of DataParallel module'''
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k, v in teacher.items():
                name = k.replace('module.', '')
                state_dict[name] = v
            net_t.load_state_dict(state_dict, strict=True)
        net_t.eval()
        net_t.to(device)
        del args_t, teacher

    if args.use_dp:
        args.gpu_ids = [i for i in range(torch.cuda.device_count())]
        net = torch.nn.DataParallel(net, device_ids=args.gpu_ids)
        if args.kd_model:
            net_t = torch.nn.parallel.DataParallel(net_t, device_ids=args.gpu_ids)
        print("Using Data Parallel")

    #####################
    # optimizer         #
    #####################
    if args.filter_bias_and_bn:
        from utils.helper_funcs import add_weight_decay
        parameters = add_weight_decay(net, args.wd)
    else:
        parameters = net.parameters()

    opt = torch.optim.AdamW(parameters,
                            lr=args.max_lr,
                            betas=[0.9, 0.99],
                            weight_decay=0,
                            eps=eps)


    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                       max_lr=args.max_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.n_epochs,
                                                       pct_start=0.1,
                                                       )

    if args.ema:
        from modules.ema import ModelEma as EMA
        ema = EMA(net, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema

    #####################
    # losses            #
    #####################
    if args.loss_type == "label_smooth":
        from modules.losses import LabelSmoothCrossEntropyLoss
        criterion = LabelSmoothCrossEntropyLoss(smoothing=0.1, reduction='sum').to(device)

    elif args.loss_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    elif args.loss_type == "focal":
        from modules.losses import FocalLoss
        criterion = FocalLoss().to(device)

    elif args.loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)

    else:
        raise ValueError

    ####################################
    # Dump arguments and create logger #
    ####################################
    root = args.save_path / args.run_name
    root.mkdir(parents=True, exist_ok=True)

    load_root = Path(args.load_path) if args.load_path else None
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    print(args)
    writer = SummaryWriter(str(root))

    #####################
    # resume training   #
    #####################
    best_acc = -1
    best_loss = 999
    steps = 0
    if load_root and load_root.exists():
        chkpnt = torch.load(load_root / "chkpnt.pt")
        try:
            net.load_state_dict(chkpnt['model_dict'], strict=True)
        except:
            '''remove module. prefix in case of DataParallel module'''
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k, v in chkpnt['model_dict'].items():
                name = k.replace('module.', '')
                state_dict[name] = v
            net.load_state_dict(state_dict, strict=True)
            del state_dict
        if args.resume_training:
            opt.load_state_dict(chkpnt['opt_dict'])
            if scaler is chkpnt.keys() and chkpnt['scaler'] is not None:
                scaler.load_state_dict(chkpnt["scaler"])
            if lr_scheduler is chkpnt.keys() and chkpnt['lr_scheduler'] is not None:
                lr_scheduler.load_state_dict(chkpnt['lr_scheduler'])
            steps = chkpnt['steps'] if 'steps' in chkpnt.keys() else 0

        best_acc = chkpnt['best_acc']
        if 'best_loss' in chkpnt.keys():
            best_loss = chkpnt['best_loss']

        print('checkpoints loaded')
    else:
        best_acc = -1
        best_loss = 999
        steps = 0
    print(best_acc)

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    dummy_run(net, args.batch_size, args.seq_len)
    costs = []
    net.train()
    start = time.time()
    skip_scheduler = False
    for epoch in range(1, args.n_epochs + 1):
        if args.use_ddp:
            sampler.set_epoch(epoch)
        t_epoch = time.time()

        if epochs_from_last_reset <= 1:  # two first epochs do ultra short-term ema
            ema.decay_per_epoch = 0.01
        else:
            ema.decay_per_epoch = decay_per_epoch_orig
        epochs_from_last_reset += 1

        # set 'decay_per_step' for the eooch
        ema.set_decay_per_step(len(train_loader))

        for iterno, (x, y) in enumerate(train_loader):
            t_batch = time.time()
            x = x.to(device)
            if args.multilabel:
                y = [F.one_hot(torch.Tensor(y_i).long(), args.n_classes).sum(dim=0).float() for y_i in y]
                y = torch.stack(y, dim=0).contiguous().to(device)
            else:
                y = y.to(device)
            x, targets, is_mixed = batch_augs(x, y, epoch)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                pred = net(x)
                if is_mixed:
                    loss_cls = batch_augs.mix_loss(pred, targets, n_classes=args.n_classes,
                                                   pred_one_hot=args.multilabel)
                else:
                    loss_cls = criterion(pred, y)

            if args.kd_model:
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    with torch.no_grad():
                        pred_t = net_t(x)
                    if args.multilabel:
                        loss_cls += F.kl_div(F.logsigmoid(pred), torch.sigmoid(pred_t), reduction='batchmean')
                    else:
                        loss_cls += F.kl_div(pred.log_softmax(-1), pred_t.softmax(-1), reduction='batchmean')
            ###################
            # Train Generator #
            ###################
            net.zero_grad()
            if args.amp:
                scaler.scale(loss_cls).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                scaler.step(opt)
                amp_scale = scaler.get_scale()
                scaler.update()
                skip_scheduler = amp_scale != scaler.get_scale()
            else:
                loss_cls.backward()
                opt.step()

            if args.ema:
                ema.update(net, steps)


            if not skip_scheduler:
                lr_scheduler.step()

            if not args.multilabel:
                acc = accuracy(pred.detach().data, y.detach().data, topk=(1,))[0]
                acc = acc.item()
            else:
                acc = mAP(y.detach().cpu().numpy(), torch.sigmoid(pred).detach().cpu().numpy())
            costs.append([acc, loss_cls.item(), opt.param_groups[0]['lr']])
            ######################
            # Update tensorboard #
            ######################
            if steps % args.log_interval == 0:
                if not args.use_ddp or (args.use_ddp and torch.distributed.get_rank() == 0):
                    writer.add_scalar("train/acc", costs[-1][0], steps)
                    writer.add_scalar("train/ce", costs[-1][1], steps)
                    writer.add_scalar("train/lr", costs[-1][2], steps)

                t_batch = time.time() - t_batch
                print("epoch {}/{} | iters {}/{} | ms/batch {:5.2f} | acc/loss {}".format(
                    epoch,
                    args.n_epochs,
                    iterno,
                    len(train_loader),
                    1000 * t_batch / args.log_interval,
                    np.asarray(costs).mean(0))
                )
                costs = []
                start = time.time()

            steps += 1
            if steps % args.save_interval == 0:
                ''' validate'''
                net.eval()
                st = time.time()
                loss = 0
                if args.multilabel:
                    labels = np.zeros((len(test_loader.dataset), args.n_classes)).astype(np.float32)
                    preds = np.zeros((len(test_loader.dataset), args.n_classes)).astype(np.float32)
                else:
                    cm = np.zeros((args.n_classes, args.n_classes), dtype=np.int32)
                idx_start = 0
                with torch.no_grad():
                    acc = 0
                    for i, (x, y) in enumerate(test_loader):
                        x = x.to(device)
                        if args.multilabel:
                            y = [F.one_hot(torch.Tensor(y_i).long(), args.n_classes).sum(dim=0).float() for y_i in y]
                            y = torch.stack(y, dim=0).contiguous().to(device)
                            y = y.to(device)
                            pred = net(x)
                            loss += F.binary_cross_entropy_with_logits(pred, y)
                            idx_end = idx_start + y.shape[0]
                            preds[idx_start:idx_end, :] = torch.sigmoid(pred).detach().data.cpu().numpy()
                            labels[idx_start:idx_end, :] = y.detach().data.cpu().numpy()
                            idx_start = idx_end
                        else:
                            y = y.to(device)
                            pred = net(x)
                            _, y_est = torch.max(pred, 1)
                            loss += F.cross_entropy(pred, y)
                            acc += accuracy(pred.detach().data, y.detach().data, topk=[1, ])[0].item()
                            for t, p in zip(y.view(-1), y_est.view(-1)):
                                cm[t.long(), p.long()] += 1
                    loss /= len(test_loader)

                    if args.multilabel:
                        acc = mAP(labels, preds)
                    else:
                        # acc /= len(test_loader)
                        acc = 100*np.diag(cm).sum()/ len(test_loader.dataset)

                writer.add_scalar("test/acc", acc, steps)
                writer.add_scalar("test/ce", loss.item(), steps)

                best_acc, best_loss = save_model(net, opt, loss, best_loss, acc, best_acc, steps, root, lr_scheduler=lr_scheduler, scaler=scaler)

                print(
                    "test: Epoch {} | Iters {} / {} | ms/batch {:5.2f} | acc/best acc/loss {:.2f} {:.2f} {:.2f} {:.2f}".format(
                        epoch,
                        iterno,
                        len(test_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        acc,
                        best_acc,
                        loss,
                        best_loss,
                    )
                )

                print("Took %5.4fs to save samples" % (time.time() - st))
                print("-" * 100)
                net.train()

        t_epoch = time.time() - t_epoch
        print("epoch {}/{} time {:.2f}".format(epoch, args.n_epochs, t_epoch / args.log_interval))


def main():
    args = parse_args()
    args = check_args(args)
    train(args)


if __name__ == "__main__":
    main()