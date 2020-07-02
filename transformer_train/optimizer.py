
import torch
import math


class Optimizer(object):
    def __init__(self, model, params, update_lr_stepwise=False, parallel_mode='dp'):

        self.params = params
        self.model = model
        self.update_lr_stepwise = update_lr_stepwise
        self.parralle_mode = parallel_mode

        self.lr = self.params['lr']
        self.global_step = 1
        self.global_epoch = 0

        if params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr,
                                              betas=(0.9, 0.98), eps=1e-9)
        elif params['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, momentum=0.9)
        elif params['optimizer'] == 'adadelate':
            self.optimizer = torch.optim.Adadelta(
                filter(lambda p: p.requires_grad, model.parameters()))
        else:
            raise NotImplementedError

        if self.parralle_mode == 'hvd':
            import horovod.torch as hvd
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=model.named_parameters())

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def set_lr(self, lr=None):
        new_lr = self.lr if lr is None else lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def step(self):
        self.optimizer.step()
        if self.update_lr_stepwise:
            self.lr = self.get_lr()
            self.set_lr()
        self.global_step += 1

    def epoch(self):
        if not self.update_lr_stepwise:
            self.lr = self.get_lr()
            self.set_lr()
        self.global_epoch += 1


class TransformerOptimizer(Optimizer):

    def __init__(self, model, params, model_size, parallel_mode='dp'):
        super(TransformerOptimizer, self).__init__(model, params, True, parallel_mode)

        self.model_size = model_size
        self.factor = params['lr']
        self.warmup_steps = params['warmup_steps']
        self.lr = self.get_lr()
        self.set_lr()

    def get_lr(self):
        return self.factor * self.model_size ** (-0.5) * min(self.global_step ** (-0.5), self.global_step * self.warmup_steps ** (-1.5))





# import os
# import sys
# import datetime
# import torch
# import math
# from torch.optim.optimizer import Optimizer

# class AdamW(Optimizer):
#     """Implements Adam algorithm.
#     It has been proposed in `Adam: A Method for Stochastic Optimization`_.
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 0.999))
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-8)
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#         amsgrad (boolean, optional): whether to use the AMSGrad variant of this
#             algorithm from the paper `On the Convergence of Adam and Beyond`_
#     .. _Adam\: A Method for Stochastic Optimization:
#         https://arxiv.org/abs/1412.6980
#     .. _On the Convergence of Adam and Beyond:
#         https://openreview.net/forum?id=ryQu7f-RZ
#     """

#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=0, amsgrad=False):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay, amsgrad=amsgrad)
#         super(AdamW, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(AdamW, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)

#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#                 amsgrad = group['amsgrad']

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p.data)
#                     if amsgrad:
#                         # Maintains max of all exp. moving avg. of sq. grad. values
#                         state['max_exp_avg_sq'] = torch.zeros_like(p.data)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 if amsgrad:
#                     max_exp_avg_sq = state['max_exp_avg_sq']
#                 beta1, beta2 = group['betas']

#                 state['step'] += 1

#                 # if group['weight_decay'] != 0:
#                 #     grad = grad.add(group['weight_decay'], p.data)

#                 # Decay the first and second moment running average coefficient
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 if amsgrad:
#                     # Maintains the maximum of all 2nd moment running avg. till now
#                     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
#                     # Use the max. for normalizing running avg. of gradient
#                     denom = max_exp_avg_sq.sqrt().add_(group['eps'])
#                 else:
#                     denom = exp_avg_sq.sqrt().add_(group['eps'])

#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#                 step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

#                 # p.data.addcdiv_(-step_size, exp_avg, denom)
#                 p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

#         return loss


# class TransformerOptimizer():
#     "Optim wrapper that implements rate."

#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0

#     def step(self):
#         "Update parameters and rate"
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer. param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()

#     def rate(self, step=None):
#         "Implement `lrate` above"
#         if step is None:
#             step = self._step
#         return self.factor * ((self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5)))

#     def zero_grad(self):
#         self.optimizer.zero_grad()

#     def save(self, path):
#         all = {'opt_state': self.optimizer.state_dict(),
#                'step': self._step, 'factor': self.factor, 'model_size': self.model_size, 'rate':self._rate}
#         torch.save(all, path)
#         print(f'opt saved to {path}')

#     def load(self, path):
#         all = torch.load(path)
#         self.optimizer.load_state_dict(all['opt_state'])
#         self._step = all['step']
#         self.factor = all['factor']
#         self.model_size = all['model_size']
#         self._rate = all['rate']
#         print(f'opt loaded from {path}')
