import os
import sys
import time
import torch
import numpy as np
import random
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
from collections import deque
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, ins, dataloader, train_batches, max_train_batches,logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['save_cycle'] * args.milestone
        # print('self.train_num_steps ',self.train_num_steps)
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
#         self.dl = cycle(dataloader['dataloader'])

        self.ins = ins 
        self.train_loaders = dataloader
        self.train_batches = train_batches
        self.max_train_batches=max_train_batches
        # print('train_batches is ',train_batches)
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path('Checkpoints_'+ args.name + f'_{model.seq_length}'+ f'_maskrate{args.mask_rate}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone



    def train(self,milestone):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        import math
        weak_dataset_idx = self.args.weak_idx
        alpha = self.args.alpha
        
        # weight_by_idx = {
        #     0: 0.93, 1: 0.95, 2: 0.94, 3: 0.97,
        #     4: 1.00, 5: 0.91, 7: 0.92,
        # }

        num_datasets = len(self.train_loaders)
        window_size = 100
        loss_histories = [deque(maxlen=window_size) for _ in range(num_datasets)]
        steps_per_epoch = math.ceil(self.train_batches / self.gradient_accumulate_every)
        total_steps      = steps_per_epoch * self.args.epoch
        with tqdm(initial=step, total=total_steps) as pbar:
            for e in range(milestone, self.args.epoch):

                print('##############')
                print('Current Epoch ', e)
                print('##############')
                # —— 仅在前两个 epoch 计算 weight_by_idx —— #
                if e < 2:
                    avg_losses = [(np.mean(h) if len(h)>0 else 1.0) for h in loss_histories]
                    max_l, min_l = max(avg_losses), min(avg_losses)
                    if max_l > min_l:
                        weight_by_idx = {
                            i: 0.9 + (max_l - l) / (max_l - min_l) * (0.97 - 0.9)
                            for i, l in enumerate(avg_losses)
                        }
                    else:
                        weight_by_idx = { i: 0.97 for i in range(num_datasets) }
                    # 在 e==1 时把第一轮和第二轮的计算结果固定下来
                    if e == 1:
                        fixed_weights = weight_by_idx.copy()
                else:
                    # 后续 epoch 直接复用
                    weight_by_idx = fixed_weights

                # 每个 epoch 重置迭代器和标记
                iterators = [dl._get_iterator() for dl in self.train_loaders]
                finished  = [False] * num_datasets
                batch_cnt = [0] * num_datasets

                # 直到所有 mini-batch 都被处理
                while sum(batch_cnt) < self.train_batches:
                    # —— 动态采样概率 —— #
                    avg_losses = [(np.mean(h) if len(h)>0 else 1.0) for h in loss_histories]
                    probs = np.array(avg_losses)
                    probs = probs / probs.sum()
                    alive = [i for i, f in enumerate(finished) if not f]
                    alive_probs = probs[alive] / probs[alive].sum()
                    idx = np.random.choice(alive, p=alive_probs)

                    # idx = random.randint(0, length - 1)  
                    instruct = self.ins[idx]

                    # —— pick weight: only apply during the first 20 epochs —— #
                    if e < 50:

                        w = weight_by_idx.get(idx, 1.0)
                    else:
                        w = 1.0

                    # 累积梯度
                    got_any = False
                    total_loss = 0.0
                    for _ in range(self.gradient_accumulate_every):
                        try:
                            data = next(iterators[idx]).to(device)
                            # —— 你的数据预处理逻辑 —— #
                            b, t, n = data.shape
                            data = data.permute(0, 2, 1).reshape(b*n, t, 1)
                            mask = torch.rand((b*n, t, 1), device=device)
                            mask[mask < self.args.mask_rate] = 0
                            mask[mask >= self.args.mask_rate] = 1
                            model_input = data.masked_fill(mask == 0, 0)

                            # —— 前向计算 + 弱势数据集加权 —— #
                            loss = self.model(instruct, model_input, mask=mask, target=model_input)
                            if idx == weak_dataset_idx:
                                loss = loss * alpha

                            loss = loss / self.gradient_accumulate_every
                            loss.backward()

                            total_loss += loss.item()
                            batch_cnt[idx] += 1
                            got_any = True
                        except StopIteration:
                            finished[idx] = True
                            break

                    if not got_any:
                        continue

                    # —— 更新参数 —— #
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sch.step(total_loss)
                    self.opt.zero_grad()
                    self.step += 1
                    step += 1
                    self.ema.update()

                    # —— 更新 loss history —— #
                    loss_histories[idx].append(total_loss)

                    # —— 日志 & checkpoint —— #
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                    pbar.update(1)
                    pbar.set_description(f'Step: {step}, Loss: {total_loss:.6f}')

                    if sum(batch_cnt) >= self.train_batches:
                        with torch.no_grad():
                            self.milestone += 1
                            if self.milestone % 1 == 0:
                                self.save(self.milestone)
                        break
                        

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, instruct, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(instruct, batch_size=size_every)
            # print('sample shape is : ',sample.shape)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples
