#生成wav，txt文件的路径文件csv

import os
import tarfile
from tqdm import tqdm
import torch
import shutil
import sys
import soundfile
import librosa
from pyvad import trim
import python_speech_features
from collections import Counter
import numpy as np
import random
import os
import csv
import pickle
from datetime import datetime
def cal_duration(file):
   sig, sr = soundfile.read(file)
   duration = len(sig) / sr
   return duration

class ExtractFeatures(object):
   def __init__(self):
      super(ExtractFeatures, self).__init__()

   def feature_from_manifist(self,data,if_augment):
        if if_augment==False:
            return data
        sig = self._aug_amplitude(data)
        sig = self._aug_speed(sig)
        feature = self._feature_mel(sig)
        feature = self._feature_lfr(feature)
        feature = self._normalize(feature)
        feature = self._aug_freq_time_mask(feature)
        feature = torch.from_numpy(feature)
        return feature
   def wav_feature(self, path, if_augment=False):
      sig = self._load_wav(path)
      sig = self._sig_vad(sig)
      if if_augment:
         #sig = sig.tostring()
         return sig
      #return sig
      if if_augment:
         sig = self._aug_amplitude(sig)
         sig = self._aug_speed(sig)
      feature = self._feature_mel(sig)
      feature = self._feature_lfr(feature)
      feature = self._normalize(feature)
      if if_augment:
         feature = self._aug_freq_time_mask(feature)
      #print(feature.type())
      #feature = torch.from_numpy(feature)
      #print(feature.type())
      #sys.exit()
      
      return feature

   def _load_wav(self, wav_file):
      sig, sample_rate = librosa.load(wav_file, sr=16000)
      return sig

   def _sig_vad(self, sig):
      tmp = sig
      if True:
         sig = trim(sig, 16000, fs_vad=16000, hoplength=30, thr=0, vad_mode=2)
      if sig is None:
         return tmp
      else:
         return sig

   def _guassian_noise(self, y, is_train=False, loc=0.0, scale=0.01):
      if is_train:
         noise = np.random.normal(loc=loc, scale=scale, size=np.shape(y))
         return y + noise
      return y

   def _aug_amplitude(self, sig):
      nsig = sig * random.uniform(0.9, 1.1)
      return nsig

   def _aug_speed(self, sig):
      speed_rate = random.Random().uniform(0.9, 1.1)
      old_length = sig.shape[0]
      new_length = int(sig.shape[0] / speed_rate)
      old_indices = np.arange(old_length)
      new_indices = np.linspace(start=0, stop=old_length, num=new_length)
      nsig = np.interp(new_indices, old_indices, sig)
      return nsig

   def _feature_mel(self, signal):
      tensor = python_speech_features.logfbank(
         signal,
         samplerate=16000,
         winlen=0.025,
         winstep=0.01,
         nfilt=80,
         lowfreq=40,
         highfreq=8000
      )
      # [L, H]
      return tensor

   def _feature_lfr(self, tensor):
      m = 4
      n = 3
      LFR_inputs = []
      T = tensor.shape[0]
      T_lfr = int(np.ceil(T / n))
      for i in range(T_lfr):
         if m <= T - i * n:
            LFR_inputs.append(np.hstack(tensor[i * n:i * n + m]))
         else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(tensor[i * n:])
            for _ in range(num_padding):
               frame = np.hstack((frame, tensor[-1]))
            LFR_inputs.append(frame)
      feature = np.vstack(LFR_inputs)
      return feature

   def _normalize(self, tensor):
      tensor = (tensor - tensor.mean()) / tensor.std()
      return tensor

   def _aug_freq_time_mask(self, tensor):
      if not isinstance(tensor, torch.Tensor):
         tensor = torch.from_numpy(tensor)
      tensor.unsqueeze_(0)
      tensor = self.freq_mask(tensor, F=27, num_masks=1, replace_with_zero=True)
      tensor_len = tensor.size(1)
      max_len = int(tensor_len * 0.2)
      T = min(max_len, 25)
      if T == 0:
         T += 1
      tensor = self.time_mask(tensor, T=T, num_masks=1, replace_with_zero=True)
      tensor.squeeze_(0)
      tensor = tensor.numpy()
      return tensor

   def freq_mask(self, spec, F=50, num_masks=2, replace_with_zero=True):
      cloned = spec.clone()
      num_mel_channels = cloned.shape[2]

      for i in range(0, num_masks):
         f = random.randrange(0, F)
         f_zero = random.randrange(0, num_mel_channels - f)

         # avoids randrange error if values are equal and range is empty
         if (f_zero == f_zero + f): return cloned

         mask_end = random.randrange(f_zero, f_zero + f)
         if (replace_with_zero):
            cloned[0][:, f_zero:mask_end] = 0
         else:
            cloned[0][:, f_zero:mask_end] = cloned.mean()

      return cloned

   def time_mask(self, spec, T=30, num_masks=1, replace_with_zero=True):
      cloned = spec.clone()
      len_spectro = cloned.shape[1]

      for i in range(0, num_masks):
         t = random.randrange(0, T)
         t_zero = random.randrange(0, len_spectro - t)

         # avoids randrange error if values are equal and range is empty
         if t_zero == t_zero + t:
            return cloned

         mask_end = random.randrange(t_zero, t_zero + t)
         if replace_with_zero:
            cloned[0][t_zero:mask_end] = 0
         else:
            cloned[0][t_zero:mask_end] = cloned.mean()

      return cloned
def build_manifist():
    data_folder='zh_data/all_test.csv'
    getfeatures = ExtractFeatures()
    with open(data_folder, encoding='utf8') as file:
        data = file.readlines()
    manifist = []
    manifist_folder = '/media/data/weijiang_hd_data/djl/zh_features/all_test.manifist'
    for i in tqdm(range(len(data)), ncols=75):
         wave_file, target = data[i].split(',', 1)
         #wav_features = getfeatures.wav_feature(wave_file, if_augment=False)
         sample = {'wav_file': wave_file,'target': target}
         #manifist.append(sample)
         duration = cal_duration(wave_file)
         if duration<=10:
            sample = {'wav_file': wave_file,'target': target}
            manifist.append(sample)
    torch.save(manifist, manifist_folder)
   #  iterm=60000
   #  iterm_size=len(data)//iterm+1
   #  print(len(data),iterm_size)
   #  for ids in range(iterm_size):
   #     manifist_folder ='/media/data/urun_tandong_video/data/djl/zh_features/all_train%s.manifist'%(ids)
   #     print(manifist_folder)
   #     start = ids*iterm
   #     end = (ids+1)*iterm
   #     if end>len(data):
   #        end=len(data)
   #     manifist = []
   #     print('start-end',start, end)
   #     for i in tqdm(range(start, end)):
   #        try:
   #           wave_file, target = data[i].split(',', 1)
   #           wav_features = getfeatures.wav_feature(wave_file, if_augment=False)
   #           duration = cal_duration(wave_file)
   #           sample = {'wav_features': wav_features,
   #                     'target': target,
   #                     'duration': duration}
   #           manifist.append(sample)
   #        except:
   #           print('error')
   #     torch.save(manifist, manifist_folder)
    print('manifist built')
def build_csv():
   data_folder = 'zh_data/all_train.csv'
   manifist_folder = '/media/data/weijiang_hd_data/djl/zh_features/all_train_10.csv'
   #getfeatures = ExtractFeatures()
   with open(data_folder, encoding='utf8') as file:
      data = file.readlines()
   f = open(manifist_folder, "w", encoding='utf-8', newline='')
   writer = csv.writer(f, dialect='excel')
   max_len=0
   for i in tqdm(range(len(data))):
      b = []
      wave_file, target = data[i].split(',', 1)
      #wav_features = getfeatures.wav_feature(wave_file, if_augment=True)
      target = target.rstrip()
      duration = cal_duration(wave_file)
      if duration<=10:
         b.append(wave_file)
         b.append(target)
         writer.writerow(b)
      else:
         if max_len<duration:
            max_len = duration
            print(max_len)
   f.close()
def build_txt():
   data_folder = 'zh_data/aishell-1_train.csv'
   manifist_folder = '/media/data/weijiang_hd_data/djl/zh_features/aishell-1_train'
   getfeatures = ExtractFeatures()
   with open(data_folder, encoding='utf8') as file:
      data = file.readlines()
   #f = open(manifist_folder, "wb")
   #manifist=[]
   if_augment=True
   time1=datetime.now()
   # paths=[]
   # samples=[]
   for i in tqdm(range(len(data))):
      wave_file, target = data[i].split(',', 1)
      wavpath, wavnames = os.path.split(wave_file)
      wavname = ''.join(os.path.splitext(wavnames)[0:-1])
      wav_features = getfeatures.wav_feature(wave_file, if_augment)
      duration = cal_duration(wave_file)
      if if_augment==False or (if_augment and duration<=10):
         sample = {'wav_features': wav_features,'target': target}
         path = os.path.join(manifist_folder, '%s.manifist'%(wavname))
         # samples.append(sample)
         # paths.append(path)
         try:
            torch.save(sample, path)
            sample = torch.load(path)
         except:
            print('error')
   # print(len(samples),len(paths))
   # for i in tqdm(range(len(paths))):
   #    try:
   #       torch.save(sample[i], path[i])
   #    except:
   #          print('error')
   # for i in tqdm(range(len(paths))):
   #    sample = torch.load(path[i])
   time2 = datetime.now()
   # files = os.listdir(manifist_folder)
   # for file in files:
   #    file_path = os.path.join(manifist_folder, file)
   #    sample = torch.load(file_path)
   # time3 = datetime.now()
   # print(sample)
   print(time2-time1)
   #pickle.dump(manifist,f)
   #f.close()

def build_npy():
   getfeatures = ExtractFeatures()
   data_folder = 'zh_data/all_test.csv'
   npy = '/media/data/weijiang_hd_data/djl/zh_features/aishell-1_test_npy.csv'
   npy_folder = '/media/data/weijiang_hd_data/djl/zh_features/aishell-1_test_npy'
   with open(data_folder, encoding='utf8') as file:
      data = file.readlines()
   f = open(npy, "w", encoding='utf-8', newline='')
   writer = csv.writer(f, dialect='excel')
   for i in tqdm(range(len(data))):
      b = []
      wave_file, target = data[i].split(',', 1)
      wav_features = getfeatures.wav_feature(wave_file, if_augment=False)
      target = target.rstrip()
      wavpath, wavnames = os.path.split(wave_file)
      wavname = ''.join(os.path.splitext(wavnames)[0:-1])
      path = os.path.join(npy_folder, '%s.npy'%(wavname))
      duration = cal_duration(wave_file)
      if duration<=10:
         np.save(path,wav_features)
         b.append(path)
         b.append(target)
         writer.writerow(b)
   f.close()

def tokenize_fn(x):
   return [i for i in x]
class Vocab:

   def __init__(self, PAD='$', UNK='%', BOS='^', EOS='&',
                tokenize_fn=tokenize_fn):
      self._counter = Counter()
      self.PAD = PAD
      self.UNK = UNK
      self.BOS = BOS
      self.EOS = EOS
      self._token2id = {v: i for i, v in enumerate([PAD, UNK, BOS, EOS]) if v is not None}
      self._id2token = None
      self._tokenize_fn = tokenize_fn

   def consume_sentance(self, sentance: str):
      sentance = self._tokenize_fn(sentance)
      self._counter.update(sentance)

   def consume_sentance_list(self, sentance_list: list):
      for sentance in sentance_list:
         self.consume_sentance(sentance)

   def build(self, min_count: int = 1, max_vocab: int = 20000):
      for i in self._counter.most_common(max_vocab):
         if i[1] >= min_count:
            self._token2id[i[0]] = len(self._token2id)

      self._id2token = [i for i in self._token2id]
      print(f'total {len(self._token2id)} words in vocab')

   def save(self, path: str):
      assert self._id2token is not None
      all = (self._id2token, self._token2id, self.PAD, self.UNK, self.BOS, self.EOS)
      torch.save(all, path)
      print(f'vocab saved in {path}')

   @classmethod
   def load(cls, path: str):
      obj = cls()
      all = torch.load(path)
      obj._id2token = all[0]
      obj._token2id = all[1]
      obj.PAD = all[2]
      obj.UNK = all[3]
      obj.BOS = all[4]
      obj.EOS = all[5]
      print(f'vocab loaded from {path}')
      return obj

   def convert_str(self, string: str, use_bos: bool = True, use_eos: bool = True):
      token = self._tokenize_fn(string)
      if use_bos:
         token = [self.BOS] + token
      if use_eos:
         token = token + [self.EOS]
      id = self.convert_token(token)
      return id

   def convert_token(self, token: list):
      id = [self._token2id.get(i, self._token2id[self.UNK]) for i in token]
      return id

   def convert_id(self, id: list):
      assert self._id2token is not None
      token = [self._id2token[i] for i in id]
      # if use_label:
      #     token = [self.BOS] + token + [self.EOS]
      return token

   def convert_id2str(self, id: list):
      assert self._id2token is not None
      token = [self._id2token[i] for i in id if i != self._token2id[self.PAD]]
      token = ' '.join(token)
      return token

   @property
   def bos_id(self):
      return self._token2id[self.BOS]

   @property
   def eos_id(self):
      return self._token2id[self.EOS]

   @property
   def pad_id(self):
      return self._token2id[self.PAD]

   @property
   def vocab_size(self):
      return len(self._token2id)
def build_vocab():
   vocab_path='/media/data/weijiang_hd_data/djl/zh_features/aishell-1_vocab_cnn.t'#空文本
   train_manifist_file = 'zh_data/aishell-1_train.csv'#音频路径，汉字
   vocab = Vocab()
   with open(train_manifist_file, encoding='utf8') as file:
      data = file.readlines()
   for i in tqdm(range(len(data))):
      try:
         wave_file, target = data[i].split(',', 1)
         vocab.consume_sentance(''.join(target))
      except:
         print('error')
   vocab.build()
   vocab.save(vocab_path)

class labels(object):
   def __init__(self,tokenize_fn=tokenize_fn):
      self._counter = Counter()
      self._token2id = {v: i for i, v in enumerate(['PAD', 'SOS', 'EOS']) if v is not None}
      self._id2token = None
      self._tokenize_fn = tokenize_fn
   def label2id(self,string: str,):
      token = self._tokenize_fn(string)
      return token
def build_labels():
   train_manifist_file = 'aidatatang_200zh_train.csv'
   train_manifist_file1 = 'aidatatang_200zh_test.csv'
   train_manifist_file2 = 'aidatatang_200zh_dev.csv'
   with open(train_manifist_file) as f:
      data = f.readlines()
   with open(train_manifist_file1) as f:
      data1 = f.readlines()
   with open(train_manifist_file2) as f:
      data2 = f.readlines()
   print(len(data),len(data1),len(data2))
   data=data+data1+data2
   #sys.exit()
   labels=[]
   for i in tqdm(range(len(data))):# 在这个字典中有就不加,没有则加,放入字典中
      wave_file, target = data[i].split(',', 1)
      for j in target:
         if j not in labels:
            labels.append(j)
      #labels = str(''.join(json.load(label_file)))
      # add PAD_CHAR, SOS_CHAR, EOS_CHAR
   print(len(labels))
   labels = ['<S/S>']+['<PAD>']+['<UNK>'] + labels
   print(len(labels))
   #sys.exit()
   label2id, id2label = {}, {}
   count = 0
   for i in range(len(labels)):
      if labels[i] not in label2id:
            label2id[labels[i]] = count
            id2label[count] = labels[i]
            count += 1
      else:
            print("multiple label: ", labels[i])
   vocab={'label2id':label2id,'id2label':id2label,'label_size':len(label2id)}
   torch.save(vocab,'aidatatang_200zh_vocab.t')
   print('save')
def open_manifist():
   time1=datetime.now()
   # train_manifist_file = '/media/data/weijiang_hd_data/djl/zh_features/all_train.csv'
   # with open(train_manifist_file) as f:
   #    data = f.readlines()
   train_manifist_file='all_vocab.t'
   data = torch.load(train_manifist_file)
   print(len(data['label2id']))
   time2=datetime.now()
   print(time2-time1)
 
if __name__== "__main__":
   #build_txt()
   #build_csv()
   #build_npy()
   #build_manifist()
   #build_vocab()
   build_labels()
   #open_manifist()
