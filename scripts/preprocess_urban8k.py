import os
import glob
import librosa
import scipy.io.wavfile as wavfile
import multiprocessing


def process_file(f, fs_trg=22050):    
    ff = f.replace("audio", "audio22_5")
    if os.path.isfile(ff):
        return True
    if os.path.getsize(f) < 100:
        return False
    x, fs = librosa.core.load(f, sr=None)
    if x.shape[0] == 0:
        return False
    if fs != fs_trg:
        x = librosa.core.resample(x, orig_sr=fs, target_sr=fs_trg)
    else:
        pass
    wavfile.write(ff, rate=fs_trg, data=x)
    return True


def resample_mp(root, fs_trg):
    fnames = glob.glob(root + '/**/*.wav', recursive=True)
    print("found ", len(fnames))    
    p = multiprocessing.Pool()
    for i, f in enumerate(fnames):
        # process_file(f, fs_trg)
        p.apply_async(process_file, [f, fs_trg])
    p.close()
    p.join()


if __name__ == '__main__':
    fs_trg = 22050
    root = '../data/UrbanSound8K'    
    resample_mp(root, fs_trg)
    print("DONE")