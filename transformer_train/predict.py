import torch
import argparse
from transformer import Transformer
from recognizer import TransformerRecognizer
from datetime import datetime
import sys

#import pycorrector

def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda: 
        checkpoint = torch.load(args.model)
    else:
        checkpoint = torch.load(args.model,map_location='cpu')
    params = checkpoint['params']
    model = Transformer(params['model'])
    model.load_state_dict(checkpoint['model'])
    print('Load model from %s' % args.model)
    if use_cuda:
        model.cuda()
    #vocab = torch.load('/home/dengjianlong/201/thchs30/vocab.t')
    #unit2char = vocab['id2label']
    unit2char = checkpoint['vocab']['id2label']
    recognizer = TransformerRecognizer(model, unit2char=unit2char, use_cuda=use_cuda)
    time1=datetime.now()
    result = recognizer.predict(args.wav)
    time2=datetime.now()
    print(time2-time1)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='ables CUDA predict')
    parser.add_argument('--model', type=str, default='/home/dengjianlong/201/pytorch-deepspeech/transformer1.2/transformer_djl/aishell/transformer/model.best.pt')
    parser.add_argument('--wav', type=str, default='/home/dengjianlong/201/test0.wav')
    cmd_args = parser.parse_args()
    main(cmd_args)
