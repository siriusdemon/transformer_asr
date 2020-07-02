import os
import yaml
import torch
import argparse
from transformer import Transformer
from recognizer import TransformerRecognizer
from data_loader import  AudioDataSet, FeatureLoader
import Levenshtein as Lev
from datetime import datetime
from tqdm import tqdm
import sys
#import pycorrector
def calculate_cer(pre, tgt):
    #s1 predicted, s2 target
    tgt = tgt.split(' ')
    tgt = ''.join(tgt)
    pre = pre.split(' ')
    pre = ''.join(pre)
    #pre, _ = pycorrector.correct(pre)
    word_num = len(tgt)
    #return Lev.distance(pre, tgt) / word_num
    return Lev.distance(pre, tgt) , word_num
def main(args):
    #path='aishell/transformer/model_best.pt'
    use_cuda = args.cuda and torch.cuda.is_available()
    checkpoint = torch.load(args.model)
    if 'params' in checkpoint:
        params = checkpoint['params']
    else:
        assert os.path.isfile(args.config), 'please specify a configure file.'
        with open(args.config, 'r') as f:
            params = yaml.load(f)
    params['data']['shuffle'] = False
    params['data']['spec_argument'] = False
    params['data']['short_first'] = False
    params['data']['batch_size'] = args.batch_size
    expdir = os.path.join( params['data']['name'], params['train']['save_name'])
    if args.suffix is None:
        decode_dir = os.path.join(expdir, 'decode_%s' % args.decode_set)
    else:
        decode_dir = os.path.join(expdir, 'decode_%s_%s' % (args.decode_set, args.suffix))

    if not os.path.exists(decode_dir):
        os.makedirs(decode_dir)

    model = Transformer(params['model'])

    model.load_state_dict(checkpoint['model'])
    print('Load pre-trained model from %s' % args.model)
    model.eval()
    if use_cuda:
        model.cuda()
    # vocab = torch.load('/home/jianlong/djl/data/thchs30/vocab.t')
    # unit2char = vocab['id2label']
    vocab = checkpoint['vocab']
    unit2char = vocab['id2label']
    dataset = AudioDataSet(params, args.test_manifist, vocab, if_augment=False)
    data_loader = FeatureLoader(dataset)
    
    # #######
    # model.eval()
    # eval_loss = 0
    # dev_bar = tqdm(enumerate(data_loader.loader), total=len(data_loader.loader), leave=True, ncols=120)
    # for step, data in dev_bar:
    #     inputs, inputs_length, targets, targets_length = data
    #     if args.ngpu > 0:
    #         inputs = inputs.cuda()
    #         targets = targets.cuda()
    #     loss = model(inputs, inputs_length, targets, targets_length)
    #     eval_loss += loss.item()
    #     desc = f'loss:{round(loss.item(), 5)}'
    #     dev_bar.set_description(desc)#显示输出信息
    # sys.exit()
    # ##########
    recognizer = TransformerRecognizer(model, unit2char=unit2char, beam_width=args.beam_width,
                    max_len=args.max_len, penalty=args.penalty, lamda=args.lamda, use_cuda=use_cuda)

    totals = len(dataset)
    batch_size = params['data']['batch_size']
    writer = open(os.path.join(decode_dir, 'predict.txt'), 'w')
    all_cer=0
    all_len=0
    all_error_num=0
    all_num=0
    test_bar = tqdm(enumerate(data_loader.loader), total=len(data_loader.loader), leave=True, ncols=120)
    for step, data in test_bar:
        inputs, inputs_length, targets, targets_length = data
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        preds = recognizer.recognize(inputs, inputs_length)
        all_len+=len(targets_length)
        cer=0
        for b in range(len(targets_length)):
            #n = step * batch_size + b
            truth = ' '.join([unit2char[i.item()] for i in targets[b][1:targets_length[b]]])
            # print('[%d / %d ]  - pred : %s' % (n, totals, preds[b]))
            # print('[%d / %d ]  - truth: %s' % (n, totals, truth))
            # writer.write( ' ' + preds[b] + '\n')
            #print(deal_cer(preds[b],truth))
            #cer+=deal_cer(preds[b],truth)
            batch_error_num, batch_num = calculate_cer(preds[b],truth)
            cer += batch_error_num/batch_num
            all_error_num += batch_error_num
            all_num += batch_num
            # cer = error_num/len()
            # all_error += error_num
            #print(cer)
        all_cer += cer
        desc = f'cer:{round(cer/len(preds), 4)},avg_cer:{round(all_cer/all_len, 4)},_cer:{round(all_error_num/all_num, 4)}'
        test_bar.set_description(desc)#显示输出信息
        
    print(all_cer/totals)
    print(all_error_num/all_num)
    #sys.exit()
    writer.write(str(datetime.now())+'\t'+ str(all_cer) + '\n')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='all.yaml')
    parser.add_argument('--cuda', type=int, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--penalty', type=float, default=0.6)
    parser.add_argument('--lamda', type=float, default=5)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--decode_set', type=str, default='test')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--model', type=str, default='/home/dengjianlong/201/pytorch-deepspeech/transformer1.2/transformer_djl/aishell/transformer/model.best.pt')
    parser.add_argument('--test_manifist', type=str, default='aishell-1_test.csv')
    cmd_args = parser.parse_args()

    main(cmd_args)
