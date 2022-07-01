import torch
import torch.nn.functional as F
import librosa
import os
import glob
import random
from datasets.audio_augs import AudioAugs


class ESCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 segment_length,
                 sampling_rate,
                 transforms=None,
                 fold_id=None,
                 ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        fnames = glob.glob(root + "/**/*.wav")
        self._get_labels(fnames)
        if mode == 'train':
            fnames = [f for f in fnames if int(os.path.basename(f).split('-')[0]) != fold_id]
        elif mode == 'test':
            fnames = [f for f in fnames if int(os.path.basename(f).split('-')[0]) == fold_id]
        else:
            raise ValueError("wrong mode")
        self.audio_files = sorted(fnames)
        self.label2idx = dict(zip(self.labels, range(len(self.labels))))
        self.transforms = AudioAugs(transforms, sampling_rate, p=0.5) if transforms is not None else None

    def _get_labels(self, f_names):
        self.labels = sorted(list(set([f.split('/')[-2] for f in f_names])))

    def __getitem__(self, index):
        fname = self.audio_files[index]
        label = fname.split('/')[-2]
        label = self.label2idx[label]
        audio, sampling_rate = librosa.core.load(fname, sr=None, mono=True)
        audio = 0.95*librosa.util.normalize(audio)
        audio = torch.from_numpy(audio).float()

        assert("sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(sampling_rate != self.sampling_rate))
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        if self.transforms is not None:
            audio = self.transforms(audio)

        return audio.unsqueeze(0), label

    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    pass