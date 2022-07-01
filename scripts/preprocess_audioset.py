import os
import glob
import librosa
import scipy.io.wavfile as wavfile
import multiprocessing
import numpy as np
import pandas as pd
import csv
import pickle
from itertools import compress


def _get_labels(root):
    f_labels = root + '/metadata/class_labels_indices.csv'
    print("labels file found") if os.path.isfile(f_labels) else print("labels file not found")
    lines = []
    with open(f_labels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            lines.append(row)
        print(f'Processed {line_count} lines.')
    labels = pd.DataFrame(lines[1:], columns=lines[0])
    return labels


def _parse_labels():
    root = '../data/audioset'
    labels_file = _get_labels(root)
    f_meta = root + '/metadata/eval_segments.csv'
    f_meta = root + '/metadata/balanced_train_segments.csv'
    f_meta = root + '/metadata/unbalanced_train_segments.csv'
    sub = 'unbalanced_train_segments_22_5'
    with open(f_meta, 'r') as f:
        lines = f.readlines()
    lines = lines[3:]

    lines = [l.strip().split(', ') for l in lines]
    for i, l in enumerate(lines):
        if i % 1000 == 0:
            print("parsing: {}/{}".format(i, len(lines)))
        l[0] = sub + '/' + 'Y' + l[0]
        l[-1] = l[-1].replace('"', '').split(',')
        labels = []
        for t in l[-1]:
            if '/m' in t or '/t' in t or '/g' in t:
                labels += [int(labels_file.loc[labels_file['mid'] == t]['index'].iloc[0])]
        l[-1] = labels
    meta_parsed = [[os.path.dirname(m[0])+'_flac/' + os.path.basename(m[0]), m[1], m[2], m[3]] for m in lines]

    with open(root + '/' + sub + '_flac.pkl', 'wb') as fp:
        pickle.dump(meta_parsed, fp)
    return meta_parsed


def change_sub_folder(root):
    subs = ['balanced_train_segments_22_5',
            'eval_segments_22_5']
            #'unbalanced_train_segments_22_5']
    for sub in subs:
        with open(root + '/' + sub + '.pkl', 'rb') as f_pkl:
            meta = pickle.load(f_pkl)
        print("loaded")
        for m in meta:
            m[0] = m[0].replace("_22_5", "_22_5_flac")
        with open(root + '/' + sub + '_flac.pkl', 'wb') as fp:
            pickle.dump(meta, fp)
        print("DONE")
    return True


def process_file(queue):
    m = queue[0]
    f_audios = queue[1]
    root = queue[2]
    f_name = os.path.basename(m[0])
    if f_name in f_audios:
        labels = m[-1]
        ff = root + '/' + m[0] + '.wav'
        if os.path.getsize(ff) > 1000 and len(labels) > 0:
            queue.put[True]
        else:
            return False


def remove_nonexist_files_mp(root):
    sub = 'unbalanced_train_segments_22_5'
    with open(root + '/unbalanced_train_segments_22_5_parsed.pkl', 'rb') as f_pkl:
        meta = pickle.load(f_pkl)
    print("loaded")
    audio_files = glob.glob(root + '/' + sub + '/*.wav')
    f_audios = set([os.path.basename(a)[:-4] for a in audio_files])
    idx_list = np.ones(len(meta)).astype(np.bool)
    p = multiprocessing.Pool()
    meta_exist = meta.copy()
    queue = multiprocessing.Queue()
    print("starting")
    for i, m in enumerate(meta):
        queue.put([m, f_audios, root])
        resp = p.apply_async(process_file, queue)
        idx_list[i] = resp
        print(resp)
    print("ending")
    meta_exist = list(compress(meta_exist, idx_list.tolist()))
    with open(root + '/' + 'unbalanced_train_segments_22_5_new.pkl', 'wb') as fp:
        pickle.dump(meta_exist, fp)
    return meta_exist


def remove_nonexist_files(root):
    sub = 'unbalanced_train_segments_22_5'
    with open(root + '/unbalanced_train_segments_22_5_parsed.pkl', 'rb') as f_pkl:
        meta = pickle.load(f_pkl)
    print("loaded")
    audio_files = glob.glob(root + '/' + sub + '/*.wav')
    f_audios = set([os.path.basename(a)[:-4] for a in audio_files])
    idx_list = np.ones(len(meta)).astype(bool)
    # meta_exist = meta.copy()
    # meta_exist = [meta[i] for i, m in enumerate(meta) if len(m[-1]) > 0 and
    #               os.path.basename(m[0]) in f_audios and
    #               os.path.isfile(root + '/' + m[0] + '.wav') and
    #               os.path.getsize(root + '/' + m[0] + '.wav') > 1000]
    for i, m in enumerate(meta):
        if i % 1000 == 0:
            print("removing: {}/{}".format(i, len(meta)))
        f_name = os.path.basename(m[0])
        if f_name in f_audios:
            labels = m[-1]
            ff = root + '/' + m[0] + '.wav'
            if not os.path.isfile(ff) or os.path.getsize(ff) < 1000 or len(labels) == 0:
                idx_list[i] = False
    print("ending")
    meta_exist = [meta[k] for k, i in enumerate(idx_list) if i == True]
    print(len(meta), len(meta_exist), idx_list.sum())
    # meta_exist = list(compress(meta, idx_list.tolist()))
    with open(root + '/unbalanced_train_segments_22_5.pkl', 'wb') as fp:
        pickle.dump(meta_exist, fp)
    return meta_exist


def remove_nonexist_files2(root):
    sub = 'balanced_train_segments_22_5_flac'
    sub = 'eval_segments_22_5_flac'
    sub = 'unbalanced_train_segments_22_5_flac'
    with open(root + '/' + sub + '_new.pkl', 'rb') as f_pkl:
        meta = pickle.load(f_pkl)
    print("loaded")
    meta_files = [os.path.basename(m[0]) for m in meta]
    meta_files_d = {os.path.basename(m[0]) : m for m in meta}
    audio_files = glob.glob(root + '/' + sub + '/*.flac')
    print("meta ", len(meta_files), "audios ", len(audio_files))
    f_audios = [os.path.basename(a).split('.')[0] for a in audio_files]
    # pdb.set_trace()
    meta_exist = [meta_files_d[fa] for fa in f_audios
                  if fa in meta_files_d.keys()
                  and len(meta_files_d[fa][-1]) > 0
                  and os.path.getsize(root + '/' + meta_files_d[fa][0] + '.flac') > 0]
    print(len(meta), len(meta_exist))
    # pdb.set_trace()
    # with open(root + '/balanced_train_segments_22_5_flac_new.pkl', 'wb') as fp:
    # with open(root + '/eval_segments_22_5_flac_new.pkl', 'wb') as fp:
    with open(root + '/unbalanced_train_segments_22_5_flac_new.pkl', 'wb') as fp:
        pickle.dump(meta_exist, fp)
    return meta_exist


def check_nonexist_files_(root):
    subs = ['balanced_train_segments_22_5',
            'eval_segments_22_5',
            'unbalanced_train_segments_22_5']

    subs = ['unbalanced_train_segments_22_5']
    for sub in subs:
        with open(root + '/' + sub + '.pkl', 'rb') as f_pkl:
            meta = pickle.load(f_pkl)
        print("meta1 ", len(meta))
        # pdb.set_trace()
        # meta_bad = []
        # cnt = 0
        # import pdb
        # pdb.set_trace()
        # meta_ = [m for m in meta if os.path.isfile(root + '/' + m[0] + '.wav') and os.path.getsize(root + '/' + m[0] + '.wav') > 0]
        meta_ = [[os.path.dirname(m[0]) + '_ogg/' + os.path.basename(m[0]), m[1], m[2], m[3]]
                 for m in meta if os.path.isfile(root + '/' + m[0] + '.wav')
                 and os.path.getsize(root + '/' + m[0] + '.wav') > 0]
        # for m in meta_:
        #     d = os.path.dirname(m[0])
        #     fn = os.path.basename(m[0])
        #     m[0] = d + '_ogg/' + fn

        with open(root + '/' + sub + '_ogg.pkl', 'wb') as f_pkl:
            pickle.dump(meta_, f_pkl)

        print(len(meta), len(meta_))
    return True


def process_file(f, root_dst, fs_trg=22050):
    dirname = os.path.dirname(f)
    ff = f.replace(dirname, root_dst)
    if os.path.isfile(ff):
        return True
    if os.path.getsize(f) < 100:
        return False
    x, fs = librosa.core.load(f, sr=None)
    if x.shape[0] == 0:
        return False
    if fs != fs_trg:
        x = librosa.core.resample(x, fs, fs_trg)
    else:
        pass
    wavfile.write(ff, rate=fs_trg, data=x)
    return True


def resample_mp(root):
    fnames = glob.glob(root + '/*.wav', recursive=True)
    print("found ", len(fnames))
    # dirname = os.path.dirname(fnames[0])
    # root_dst = dirname + '_22_5'
    root_dst = '../data/audioset/unbalanced_train_segments_22_5'
    if not os.path.isdir(root_dst):
        os.mkdir(root_dst)
    p = multiprocessing.Pool()
    for i, f in enumerate(fnames):
        p.apply_async(process_file, [f, root_dst])
    p.close()
    p.join()


if __name__ == '__main__':
    # root = '../data/ESC/ESC-50'
    # root = '../data/audioset_tagging_cnn/datasets/audioset201906/audios/balanced_train_segments'
    # root = '../data/UrbanSound8K/audio'
    # root = '../data/audioset/eval_segments/eval_segments'
    # resample(root)
    # resample_mp(root)
    # compare_files('../data/audioset/balanced_train_segments/balanced_train_segments',
    #               '../data/audioset/balanced_train_segments/balanced_train_segments_22_5')
    root_base = '../data/audioset/unbalanced/audios/unbalanced_train_segments/unbalanced_train_segments_part'
    for k in range(10):
        if k != 9:
            continue
        root = root_base + str(k).zfill(2)
        print(root)
        # path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        # for file in files:
        #     print(len(path) * '---', file)
        resample_mp(root)
        print(root)