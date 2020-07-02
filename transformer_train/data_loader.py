import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
#import librosa
#from pyvad import trim
#import python_speech_features
import numpy as np
import random
import os
#from tempfile import NamedTemporaryFile
from prefetch_generator import BackgroundGenerator
#iimport kaldiio as kio
import sys
#import soundfile
from tqdm import tqdm
from datetime import datetime
import torchaudio as ta
EOS = 0
BOS = 0
PAD = 1
UNK = 2
MASK = 2
unk = '<UNK>'
compute_fbank = ta.compliance.kaldi.fbank

class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                k = k.to(device='cuda', non_blocking=True)
                # if k != 'meta':
                #     self.batch[k] = self.batch[k].to(device='cuda', non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ExtractFeatures(object):
    def __init__(self):
        super(ExtractFeatures, self).__init__()
    def wav_feature(self, path, if_augment=False):
        feature = self._load_wav(path)
        #sig = self._sig_vad(sig)
        #if if_augment:
            #sig = self._aug_amplitude(sig)
            #sig = self._aug_speed(sig)
        feature = self._fbank(feature, self.params['data']['num_mel_bins'])#40
        if if_augment:
            feature = self.spec_augment(feature)
        feature = self._normalize(feature)
        return feature
    def _load_wav(self, wav_file):
        #sig, sample_rate = librosa.load(wav_file, sr=self.config.sample_rate)
        #sig, sr=soundfile.read(wav_file)
        feature, _ = ta.load_wav(wav_file)
        return feature
    def _load_augment_wav(self, wav_file):
        tempo_value = np.random.uniform(self.config.tempo_min, self.config.tempo_max)
        gain_value = np.random.uniform(self.config.gain_min, self.config.gain_max)
        with NamedTemporaryFile(suffix=".wav") as augmented_file:
            augmented_filename = augmented_file.name
            sox_augment_params = ["tempo", "{:.3f}".format(tempo_value), "gain", "{:.3f}".format(gain_value)]
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1"\
                .format(wav_file, self.config.sample_rate, augmented_filename, " ".join(sox_augment_params))
            os.system(sox_params)
            sig = self._load_wav(augmented_filename)
            return sig
    def _sig_vad(self,sig):
        tmp = sig
#        if self.config.use_vad:
#            sig = trim(sig, 16000, fs_vad=16000, hoplength=30, thr=0, vad_mode=2)
        if sig is None:
            return tmp
        else:
            return sig
    def _aug_amplitude(self,sig):
        nsig = sig*random.uniform(0.9, 1.1)
        return nsig
    def _aug_speed(self, sig):
        speed_rate = random.Random().uniform(0.9, 1.1)
        old_length = sig.shape[1]
        new_length = int(sig.shape[1] / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sig[0] = np.interp(new_indices, old_indices, sig[0])
        return nsig
    def _fbank(self,feature,num_mel_bins):
        feature = compute_fbank(feature, num_mel_bins=num_mel_bins)
        return feature
    def spec_augment(self, feature, frequency_mask_num=1, time_mask_num=2,
                 frequency_masking_para=27, time_masking_para=15):
        tau = feature.shape[0]
        v = feature.shape[1]
        warped_feature = feature
        # Step 2 : Frequency masking
        if frequency_mask_num > 0:
            for i in range(frequency_mask_num):
                f = np.random.uniform(low=0.0, high=frequency_masking_para)
                f = int(f)
                f0 = random.randint(0, v-f)
                warped_feature[:, f0:f0+f] = 0
        # Step 3 : Time masking
        if time_mask_num > 0:
            for i in range(time_mask_num):
                t = np.random.uniform(low=0.0, high=time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau-t)
                warped_feature[t0:t0+t, :] = 0
        return warped_feature
    def _normalize(self, feature):
        feature = (feature - feature.mean()) / feature.std()
        return feature
    def wav_word(self, word_str):
        target = [self.vocab['label2id'][i] for i in word_str]
        return target

class AudioDataSet(Dataset):
    def __init__(self, params, manifist_file_dir,vocab, if_augment=False):
        super(AudioDataSet, self).__init__()

        with open(manifist_file_dir, 'r', encoding='utf8') as f:
            ids = f.readlines()
        self.params=params
        self.datas = [x.strip().split(',',1) for x in ids]
        #self.datas=self.datas[:1000]
        self.vocab = vocab
        self.if_augment=if_augment
        del ids
    def wav_feature(self, path, if_augment=False):
        feature = self._load_wav(path)
        feature = self._fbank(feature, self.params['data']['num_mel_bins'])#40
        if if_augment:
            feature = self.spec_augment(feature)
        feature = self._normalize(feature)
        return feature
    def _load_wav(self, wav_file):
        feature, _ = ta.load_wav(wav_file)
        return feature
    def _fbank(self,feature,num_mel_bins):
        feature = compute_fbank(feature, num_mel_bins=num_mel_bins)
        return feature
    def spec_augment(self, feature, frequency_mask_num=1, time_mask_num=2,
                 frequency_masking_para=27, time_masking_para=15):
        tau = feature.shape[0]
        v = feature.shape[1]
        warped_feature = feature
        # Step 2 : Frequency masking
        if frequency_mask_num > 0:
            for i in range(frequency_mask_num):
                f = np.random.uniform(low=0.0, high=frequency_masking_para)
                f = int(f)
                f0 = random.randint(0, v-f)
                warped_feature[:, f0:f0+f] = 0
        # Step 3 : Time masking
        if time_mask_num > 0:
            for i in range(time_mask_num):
                t = np.random.uniform(low=0.0, high=time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau-t)
                warped_feature[t0:t0+t, :] = 0
        return warped_feature
    def _normalize(self, feature):
        feature = (feature - feature.mean()) / feature.std()
        return feature
    def wav_word(self, word_str):
        target = [self.vocab['label2id'][i] for i in word_str]
        return target
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        sample = self.datas[item]
        feature = self.wav_feature(sample[0],if_augment=self.if_augment)
        target = self.wav_word(sample[1])
        return feature, feature.shape[0], target, len(target)

    @property
    def batch_size(self):
        return self.params['data']['batch_size']

def collate_fn_train(batch):
    features_length = [data[1] for data in batch]
    targets_length = [data[3]+1 for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    features = []
    targets = []

    for feat, feat_len, target, target_len in batch:
        features.append(np.pad(feat, 
        ((0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        targets.append([BOS] + target + [EOS] + [PAD] * (max_target_length - target_len - 1))

    features = torch.FloatTensor(features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(targets)
    targets_length = torch.IntTensor(targets_length)
    return features, features_length, targets, targets_length

def collate_fn(batch):
    features_length = [data[1] for data in batch]
    targets_length = [data[3] for data in batch]
    max_feature_length = max(features_length)
    max_target_length = max(targets_length)

    padded_features = []
    padded_targets = []

    for feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, 
        ((0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(target + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(features_length)
    targets = torch.LongTensor(padded_targets)
    targets_length = torch.IntTensor(targets_length)

    return features, features_length, targets, targets_length

class FeatureLoader(object):
    def __init__(self, dataset, shuffle=False, ngpu=1, mode='ddp'):
        if ngpu > 1:
#             if mode == 'hvd':
#                 import horovod.torch as hvd
#                 self.sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(),
#                                                                                rank=hvd.rank())
            if mode == 'ddp':
                self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                self.sampler = None
        else:
            self.sampler = None

        self.loader = DataLoaderX(dataset, batch_size=dataset.batch_size * ngpu,
                                                  shuffle=shuffle if self.sampler is None else False,
                                                  num_workers=2 * ngpu, pin_memory=True, sampler=self.sampler,
                                                  collate_fn=collate_fn_train)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
