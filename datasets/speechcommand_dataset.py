import torch
import torchaudio
import torch.nn.functional as F
import os
import glob
import random
from datasets.audio_augs import AudioAugs

sep = os.path.sep

class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 segment_length,
                 sampling_rate,
                 transforms=None,
                 use_background=False
                 ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self._get_labels(root)
        if mode == 'train':
            fnames = self.get_training_list(root)
        elif mode == 'val':
            fnames = self.load_meta_file(root, f"{sep}validation_list.txt")
        elif mode == 'test':
            fnames = self.load_meta_file(root, f"{sep}testing_list.txt")
        else:
            raise ValueError
        self.audio_files = sorted(fnames)
        self.label2idx = dict(zip(self.labels, range(len(self.labels))))
        self.transforms = transforms
        self.use_background = use_background
        if self.use_background:
            self.bg_aug = glob.glob(root + f"{sep}_background_noise_{sep}*.wav")
            self.bg_aug = [f for f in self.bg_aug if 'noise' not in os.path.basename(f)]
            self.bg_aug = [torchaudio.load(f)[0][0].detach() for f in self.bg_aug]
            self.bg_aug = [x for x in self.bg_aug]

    def load_meta_file(self, root, f_meta):
        filepath = root + f_meta
        with open(filepath) as fileobj:
            fnames = [os.path.join(root, line.strip()) for line in fileobj if line[-4:] == 'wav\n']
        return fnames

    def _get_labels(self, root):
        f_names = glob.glob(root + f"{sep}**{sep}*.wav")
        self.labels = sorted(list(set([f.split(f'{os.path.sep}')[-2] for f in f_names])))
        self.labels = sorted([l for l in self.labels if l != '_background_noise_'])

    def __getitem__(self, index):
        fname = self.audio_files[index]
        if '/' in fname:
            fname = fname.replace('/',sep)
        label = fname.split(f'{sep}')[-2]
        label = self.label2idx[label]
        audio, sampling_rate = torchaudio.load(fname)
        audio.squeeze_()
        audio = 0.95 * (audio / audio.__abs__().max()).float()

        assert ("sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(
            sampling_rate != self.sampling_rate))
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        if self.use_background and random.random() < 0.5:
            i = random.randint(0, len(self.bg_aug) - 1)
            s_level = (audio ** 2).mean()
            bg = self.bg_aug[i]
            if bg.shape[0] >= self.segment_length:
                max_bg_start = bg.size(0) - self.segment_length
                bg_start = random.randint(0, max_bg_start)
                bg = bg[bg_start: bg_start + self.segment_length]
            else:
                bg = F.pad(
                    bg, (0, self.segment_length - bg.size(0)), "constant"
                ).data
            bg_level = (bg ** 2).mean().sqrt()
            snr_db = 20 + random.random() * 5
            sgm = s_level * 10 ** (-snr_db / 10)
            bg /= bg_level
            audio.add_(bg * sgm)

        if self.transforms is not None:
            audio = AudioAugs(self.transforms, sampling_rate, p=0.5)(audio)
        return audio.unsqueeze(0), label

    def __len__(self):
        return len(self.audio_files)

    def get_training_list(self, root):
        f_names = glob.glob(root + f"{sep}**{sep}*.wav")
        f_names = [f for f in f_names if os.path.basename(os.path.dirname(f)) != '_background_noise_']
        val = self.load_meta_file(root, f"{sep}validation_list.txt")
        test = self.load_meta_file(root, f"{sep}testing_list.txt")
        valtest = val + test
        train = list(set(f_names) - set(valtest))
        return train


if __name__ == "__main__":
    pass
