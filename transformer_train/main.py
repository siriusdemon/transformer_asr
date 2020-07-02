import os
import yaml
import torch
import argparse
import sys
from transformer import Transformer
from optimizer import *
from utils import data_to_cuda
from train import Trainer
from data_loader import AudioDataSet,collate_fn
from torch.utils.data import Dataset, DataLoader
import time
import math
from tqdm import tqdm
#import pdb
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    with open(args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    expdir = os.path.join(params['data']['name'], params['train']['save_name'])
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    #load label
    #pdb.set_trace()
    vocab = torch.load(params['data']['vocab'])

    #load_data
    train_dataset = AudioDataSet(params, params['data']['train'], vocab, if_augment=True)
    dev_dataset = AudioDataSet(params, params['data']['dev'],vocab, if_augment=False)
    
    # build model
    model = Transformer(params['model'])
    if args.ngpu >= 1:
        model.cuda()

    # build optimizer
    optimizer = TransformerOptimizer(model, params['train'],
                             model_size=params['model']['d_model'],parallel_mode=args.parallel_mode)

    #class
    trainer = Trainer(params, model=model, optimizer=optimizer, vocab=vocab, is_visual=True, expdir=expdir, ngpu=args.ngpu, 
                parallel_mode=args.parallel_mode, local_rank=args.local_rank, continue_from=args.continue_from)
    
    #train test save_model
    trainer.train(train_dataset,dev_dataset)
    
  
    # print('train...')
    # model.train()
    # for step, data in tqdm(enumerate(train_loader)):
    #     inputs, inputs_length, targets, targets_length = data
    #     if args.ngpu >= 1: 
    #         inputs = inputs.cuda()
    #         targets = targets.cuda()
    #     loss = model(inputs, inputs_length, targets, targets_length)
    #     loss = torch.mean(loss) 
    #     print(loss)
    #     loss.backward()
    #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params['train']['clip_grad'])
    #     if math.isnan(grad_norm):
    #         print('Grad norm is NAN. DO NOT UPDATE MODEL!')
    #     else:
    #         optimizer.step()
    #     optimizer.zero_grad()
    
    # print('test...')
    #test

    #save model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='aidatatang_200zh.yaml')
    parser.add_argument('--ngpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--parallel_mode', type=str, default='dp')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--continue-from', dest='continue_from', action='store_true')
    cmd_args = parser.parse_args()

    main(cmd_args)
