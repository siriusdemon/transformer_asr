import torch
from asrapp.transformer import Transformer
from asrapp.recognizer import TransformerRecognizer
from datetime import datetime
import subprocess
import os
#model_path='/home/jianlong/djl/pytorch-deepspeech/transformer1.2/transformer_djl/aishell/transformer/model_best.pt'
model_path='/home/dengjianlong/201/pytorch-deepspeech/transformer1.2/transformer_djl/all/transformer/model.best.pt'
use_cuda = torch.cuda.is_available()
print('use_cuda:',use_cuda)
checkpoint = torch.load(model_path)
params = checkpoint['params']
model = Transformer(params['model'])
model.load_state_dict(checkpoint['model'])
print('Load model from %s' % model_path)
if use_cuda:
    model.cuda()
#vocab = torch.load('/home/jianlong/djl/data/thchs30/vocab.t')
#unit2char = vocab['id2label']
unit2char = checkpoint['vocab']['id2label']
recognizer = TransformerRecognizer(model, unit2char=unit2char, use_cuda=use_cuda)
def Recognition(wav_path):
    time1=datetime.now()
    new_file=wav_path
    exist_audio=False
    if wav_path[-3:]!='wav':
        new_file=wav_path[:-3]+"wav"
        command='ffmpeg -y -i %s %s'%(wav_path,new_file)
        subprocess.call(command,shell=True)
        exist_audio=True
    result = recognizer.predict(new_file)
    if exist_audio:
        os.remove(new_file)
    time2=datetime.now()
    print('used time: ',time2-time1)
    response_data = {"status": 'ok', "message": 'right', "result": result}
    return result


