
import os
import torch
import math
import time
import torch.distributed as dist
from utils import MeanLoss, init_logger, Visulizer, AverageMeter, Summary, map_to_cuda
from data_loader import FeatureLoader, DataPrefetcher
from tqdm import tqdm
import sys

class Trainer(object):
    def __init__(self, params, model, optimizer,vocab, scheduler=None, is_visual=True, expdir='./',
                 ngpu=1, parallel_mode='dp', local_rank=0, continue_from=False, opt_level='O1'):

        self.params = params
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.expdir = expdir
        self.is_visual = is_visual

        self.ngpu = ngpu
        self.parallel_mode = parallel_mode
        self.local_rank = local_rank

        self.shuffle = params['train']['shuffle']
        self.accum_steps = params['train']['accum_steps']
        self.grad_noise = params['train']['grad_noise']
        self.grad_clip = params['train']['clip_grad']
        self.global_step = 0
        self.start_epoch = 0
        self.global_loss = None
        self.log_interval = 10
        self.mean_loss = MeanLoss()
        self.vocab = vocab
        self.mixed_precision = params['train']['mixed_precision']
        self.opt_level = opt_level

        self.logger = init_logger(log_file=os.path.join(expdir, 'train.log'))
        if self.is_visual and local_rank == 0:
            self.visulizer = Visulizer(log_dir=os.path.join(expdir, 'visual'))

        if continue_from:
            model_path = os.path.join(self.expdir,self.params['train']['model_path'])
            self.load_model(model_path)
            self.logger.info('Load the checkpoint from %s' % model_path)

        if self.mixed_precision:
            import apex.amp as amp
            self.model, self.optimizer.optimizer = amp.initialize(self.model, self.optimizer.optimizer, opt_level=self.opt_level)

        if self.ngpu > 1:
#             if self.parallel_mode == 'hvd':
#                 import horovod.torch as hvd
#                 hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
#                 self.logger.info('[Horovod] Use %d gpus for training!' % self.ngpu)

            if self.parallel_mode == 'ddp':
                import torch.distributed as dist
                dist.init_process_group(backend="nccl", init_method='env://',
                                        rank=local_rank, world_size=self.ngpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
                self.logger.info('[DDP] Use %d gpus for training!' % self.ngpu)

            elif self.parallel_mode == 'dp':
                self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(self.ngpu)])
                self.logger.info('[DP] Use %d gpus for training!' % self.ngpu)

            else:
                self.logger.warning('Please chose one of dp, ddp and hvd for parallel computing!')
        elif self.ngpu == 1:
            self.logger.info('Use only 1 gpu for training!')
        else:
            self.logger.info('Train the model in CPU!')
    def train(self, train_dataset, dev_dataset):

        train_loader = FeatureLoader(train_dataset, shuffle=self.shuffle, ngpu=self.ngpu,
                                     mode=self.parallel_mode)
        dev_loader = FeatureLoader(dev_dataset, shuffle=False, ngpu=self.ngpu,
                                    mode=self.parallel_mode)
        epochs = self.params['train']['epochs']   
        TrainLossNote = Summary()
        DevLossNote = Summary()
        for epoch in range(self.start_epoch, epochs):

            self.optimizer.epoch()
            if self.parallel_mode == 'ddp':
                train_loader.set_epoch(epoch)
                self.logger.info('Set the epoch of train sampler as %d' % epoch)
            #trian
            train_loss = self.train_one_epoch(epoch, train_loader.loader)
            TrainLossNote.update(epoch, train_loss)

            if self.local_rank == 0:
                self.logger.info('-*Train-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, train_loss))
                self.save_model(epoch, loss=train_loss,save_name = 'model.train_best.pt')
            if self.is_visual and self.local_rank == 0:
                self.visulizer.add_scalar('train_epoch_loss', train_loss, epoch)
            
            #test
            dev_loss = self.test(dev_loader.loader)
            DevLossNote.update(epoch, dev_loss)
            
            #save_model
            if self.local_rank == 0:
                self.logger.info('-*Test-Epoch-%d/%d*-, AvgLoss:%.5f' % (epoch, epochs, dev_loss))     
            if dev_loss <= DevLossNote.best()[1] and self.local_rank == 0:
                self.save_model(epoch,loss=dev_loss,save_name='model.best.pt')
                self.logger.info('Update the best checkpoint!')

        if self.local_rank == 0:
            self.logger.info('Training Summary:')
            BEST_T_EPOCH, BEST_T_LOSS = TrainLossNote.best()
            self.logger.info('At the %d-st epoch of training, the model performs best (Loss:%.5f)!' % (BEST_T_EPOCH, BEST_T_LOSS))
            if dev_dataset is not None:
                BEST_E_EPOCH, BEST_E_LOSS = DevLossNote.best()
                self.logger.info('At the %d-st epoch of validation, the model performs best (Loss:%.5f)!' % (BEST_E_EPOCH, BEST_E_LOSS))
            if self.is_visual:
                self.visulizer.close()

    def train_one_epoch(self, epoch, train_loader):
        torch.cuda.empty_cache()  
        self.model.train()
        step_loss = AverageMeter()
        prefetcher = DataPrefetcher(train_loader, self.optimizer)
        train_bar = tqdm(range(len(train_loader)), leave=True, ncols=120)
        for step in train_bar:
            inputs, inputs_length, targets, targets_length = prefetcher.next()
        # train_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ncols=120)
        # for step, data in train_bar:
        #     inputs, inputs_length, targets, targets_length = data
            if self.ngpu > 0:
                inputs = inputs.cuda()
                targets = targets.cuda()   
            loss = self.model(inputs, inputs_length, targets, targets_length)      
            loss = torch.mean(loss) / self.accum_steps
            if self.mixed_precision:
                import apex.amp as amp
                with amp.scale_loss(loss, self.optimizer.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() 
            if self.grad_noise:
                raise NotImplementedError
            if self.get_rank() == 0:
                step_loss.update(loss.item() * self.accum_steps, inputs.size(0))
            if step % self.accum_steps == 0:
                # if self.local_rank == 0:
                #     self.mean_loss.update(step_loss.avg)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if math.isnan(grad_norm):
                    self.logger.warning('Grad norm is NAN. DO NOT UPDATE MODEL!')
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.is_visual and self.local_rank == 0:
                    self.visulizer.add_scalar('train_loss', loss.item(), self.global_step)
                    self.visulizer.add_scalar('lr', self.optimizer.lr, self.global_step)
                if self.global_step % self.log_interval == 0 and self.local_rank == 0:
                    #end = time.process_time()
                    # process = step * self.world_size / batch_steps * 100
                    # self.logger.info('-Training-Epoch-%d(%.2f%%), Global Step:%d, lr:%.8f, Loss:%.5f, AvgLoss: %.5f, '
                    #                     'Run Time:%.3f' % (epoch, process, self.global_step, self.optimizer.lr,
                    #                     ``                step_loss.avg, self.mean_loss.mean(), end - start))
                    #desc = f'epoch:{epoch},lr:{round(self.optimizer.lr,8)},loss:{round(loss.item(), 5)}'
                    desc = f'epoch:{epoch},lr:{round(self.optimizer.lr,8)},avg_loss:{round(step_loss.avg, 5)},loss:{round(step_loss.val, 5)}'
                    train_bar.set_description(desc)#显示输出信息
            self.global_step += 1
            #del loss, inputs, inputs_length, targets, targets_length
            #step_loss.reset()
        return  step_loss.avg
        #return self.mean_loss.mean()

    def test(self, dev_loader):
        torch.cuda.empty_cache()
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            prefetcher = DataPrefetcher(dev_loader, self.optimizer)
            dev_bar = tqdm(range(len(dev_loader)), leave=True, ncols=120)
            for step in dev_bar:
                inputs, inputs_length, targets, targets_length = prefetcher.next()
            # dev_bar = tqdm(enumerate(dev_loader), total=len(dev_loader), leave=True, ncols=120)
            # for step, data in dev_bar:
            #     inputs, inputs_length, targets, targets_length = data
                if self.ngpu > 0:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                loss = self.model(inputs, inputs_length, targets, targets_length)
                loss = torch.mean(loss)
                eval_loss += loss 
                desc = f'loss:{round(loss.item(), 5)}'
                dev_bar.set_description(desc)#显示输出信息
            eval_loss_average = eval_loss/(step+1)
            if self.global_loss is None or self.global_loss < eval_loss_average:
                self.global_loss = eval_loss_average
            
            return eval_loss_average

    def save_model(self, epoch=None,loss=None, save_name=None):
        if save_name is None:
            save_name = 'model.epoch.%d.pt' % epoch
        if self.mixed_precision:
            import apex.amp as amp
            amp_state_dict = amp.state_dict()
        else:
            amp_state_dict = None
        checkpoint = {
            'epoch': epoch+1,
            'params': self.params,
            'model': self.model.module.state_dict() if self.ngpu > 1 else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp_state_dict,
            'vocab': self.vocab,
            'loss': loss,
        }
        torch.save(checkpoint, os.path.join(self.expdir, save_name))
        self.logger.info('Save the model to %s!'%(os.path.join(self.expdir, save_name)))

    def load_model(self, checkpoint):
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.params = state_dict['params']
        self.vocab = state_dict['vocab']
        self.start_epoch = state_dict['epoch']
        self.global_loss = state_dict['loss']
        if self.mixed_precision:
            import apex.amp as amp
            amp.load_state_dict(state_dict['amp'])

    def get_rank(self):
        if self.parallel_mode == 'ddp':
            return dist.get_rank()
#         elif self.parallel_mode == 'hvd':
#             return hvd.rank()
        else:
            return 0
    @property
    def world_size(self):
        if self.ngpu > 1:
            return self.ngpu
        else:
            return 1