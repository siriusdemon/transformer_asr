import os
import csv
import sys
from tqdm import tqdm
import json
import io
import numpy as np
import os, shutil

def get_path_file(dire, file_list):
   file_or_dir = os.listdir(dire)
   dir_list =[]
   for dir_file in file_or_dir:
      #获取目录或者文件路径
      dir_file_path = os.path.join(dire, dir_file)
      if os.path.isdir(dir_file_path):
         dir_list.append(dir_file_path)
         get_path_file(dir_file_path, file_list)
      else:
         file_list.append(dir_file_path)


def aishell(file_list, content, txtname):
   a = []
   c = []
   for file in file_list:
      if os.path.splitext(file)[1] == '.wav':
         file_path = file
         a.append(str(file_path))
   # 排序
   a.sort()
   zidian1={}
   for i in range(len(content)):        
      name, hanci = content[i].split(' ', 1)
      zidian1[name] = hanci.strip().replace(' ','')
   f = open(txtname, "w", encoding='utf-8', newline='')
   writer = csv.writer(f, dialect='excel')
   num = 0
   for i in tqdm(range(len(a))):
      b = []
      b.append(a[i])
      wavpath, wavname = os.path.split(a[i])
      wavname = ''.join(os.path.splitext(wavname)[0:-1])
      if wavname in zidian1:
         b.append(zidian1[wavname])
         num += 1
         writer.writerow(b)
   if len(a) == num:
      print('全部录入', num)
   else:
      print('data length :', len(a))
      print('num length :', num)
   f.close()

if __name__== "__main__":
   data_dir='train'
   dire = "/media/psdz/data3/pub_data/data/urun_tandong_video/data/aidatatang_200zh/corpus/%s/"% data_dir  #  aishell-1 wav文件路径
   file_list = []
   get_path_file(dire, file_list)
   with open('/media/psdz/data3/pub_data/data/urun_tandong_video/data/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt', 'r') as ff:
      content = ff.readlines()
   txtname = "aidatatang_200zh_%s.csv"% data_dir # 保存位置
   aishell(file_list, content,txtname)



