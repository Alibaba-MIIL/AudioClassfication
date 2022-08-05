import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import os
import random
from datasets.audio_augs import AudioAugs
import pandas as pd
import csv
import pickle


class AudioSetDataset(torch.utils.data.Dataset):
    def __init__(self, root,
                 mode,
                 data_subtype,
                 segment_length,
                 sampling_rate,
                 transforms=None,
                 filetype='.flac',
                 calc_sample_weight=True):
        self.sampling_rate = sampling_rate
        self._get_labels(root)
        self.segment_length = segment_length
        self.root = root
        self.data_subtype = data_subtype
        self.filetype = filetype
        f_pkl = []
        if mode == 'train':
            if data_subtype == 'balanced':
                if filetype == '.wav':
                    f_pkl = [root + '/balanced_train_segments_22_5.pkl']
                elif filetype == '.ogg':
                    f_pkl = [root + '/balanced_train_segments_22_5_ogg.pkl']
                elif filetype == '.flac':
                    f_pkl = [root + '/balanced_train_segments_22_5_flac.pkl']
                else:
                    raise ValueError("bad filetype")

            elif data_subtype == 'unbalanced':
                if filetype == '.wav':
                    f_pkl = [root + '/unbalanced_train_segments_22_5.pkl']
                elif filetype == '.ogg':
                    f_pkl = [root + '/unbalanced_train_segments_22_5_ogg.pkl']
                elif filetype == '.flac':
                    f_pkl = [root + '/unbalanced_train_segments_22_5_flac.pkl']
                else:
                    raise ValueError("bad filetype")

            elif data_subtype == 'full':
                if filetype == '.wav':
                    f_pkl = [root + '/unbalanced_train_segments_22_5.pkl',
                             root + '/balanced_train_segments_22_5.pkl']
                elif filetype == '.ogg':
                    f_pkl = [root + '/unbalanced_train_segments_22_5_ogg.pkl',
                             root + '/balanced_train_segments_22_5_ogg.pkl']
                elif filetype == '.flac':
                    f_pkl = [root + '/unbalanced_train_segments_22_5_flac.pkl',
                             root + '/balanced_train_segments_22_5_flac.pkl']
                else:
                    raise ValueError("bad filetype")

        elif mode == 'test':
            calc_sample_weight = None
            if filetype == '.wav':
                f_pkl = [root + '/eval_segments_22_5.pkl']
            elif filetype == '.ogg':
                f_pkl = [root + '/eval_segments_22_5_ogg.pkl']
            elif filetype == '.flac':
                f_pkl = [root + '/eval_segments_22_5_flac.pkl']
            else:
                raise ValueError("bad filetype")
        else:
            raise ValueError

        self.meta = []
        print(f_pkl)
        for fp in f_pkl:
            with open(fp, 'rb') as f:
                self.meta += pickle.load(f)
                print(len(self.meta))

        if calc_sample_weight:
            f_sw = root + '/class_samples_' + data_subtype + '.pt'
            if os.path.isfile(f_sw):
                self.samples_weight = torch.load(f_sw)
            else:
                self.samples_weight = torch.zeros(self.__len__()).double()
                class_sample_count = self._get_stats()  # dataset has
                weights = 1 / torch.Tensor(class_sample_count.astype(np.float32)).double()
                for i, m in enumerate(self.meta):
                    if i % 10000 == 0:
                        print("calculating class sample weight: {}/{}".format(i, len(self.meta)))
                    y = m[-1]
                    self.samples_weight[i] = weights[y].sum()
                torch.save(self.samples_weight, root + '/class_samples_' + data_subtype + '.pt')

        self.transforms = transforms

    def _get_labels(self, root):
        f_labels = root + '/metadata/class_labels_indices.csv'
        if os.path.isfile(f_labels):
            print("labels file found")
        else:
            print("labels file not found")
        lines = []
        with open(f_labels) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                lines.append(row)
            print(f'Processed {line_count} lines.')
        self.labels = pd.DataFrame(lines[1:], columns=lines[0])

    def __getitem__(self, index):
        data = self.meta[index]
        fname = self.root + '/' + data[0] + self.filetype
        labels = data[-1]
        audio, sampling_rate = torchaudio.load(fname)
        audio.squeeze_()
        audio = 0.95 * (audio / audio.__abs__().max()).float()
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        if self.transforms is not None:
            audio = AudioAugs(self.transforms, sampling_rate, p=0.5)(audio)
        return audio.unsqueeze(0), labels

    def __len__(self):
        return len(self.meta)


if __name__ == "__main__":
    pass