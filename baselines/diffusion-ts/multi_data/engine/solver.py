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

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        # 改这里    
        import math
        with tqdm(initial=step, total=self.train_batches*self.args.epoch/2) as pbar:
            for e in range(self.args.epoch):
                
                print('train_loaders: ',self.train_loaders)
                iterators = [d._get_iterator() for d in self.train_loaders]
                import itertools
                for idx, iterator in enumerate(iterators):
                    print(f"Iterator {idx}: {iterator}")
                    print(f"Type: {type(iterator)}")
                length = len(self.train_loaders)
                print('length ', length)
                batch_cnt = [0] * length            
                # train
                # t1 = time.time()
                train_loss = []         
                
                exhausted = False  # 标志变量
                    # print('idx is: ', idx)
                while True:
                    # if exhausted:
                    #     print('')
                    #     print(f"Epoch {e+1}: Exiting due to StopIteration.")
                    #     break  # 跳出 while True，进入下一个 epoch                                  
                    total_loss = 0.     
                    idx = random.randint(0, length - 1)                                        
                    
                    for _ in range(self.gradient_accumulate_every):
                        try:
                            loader = iterators[idx]
                            instruct = self.ins[idx]
                            data = next(loader).to(device)
                            # print(f"loading {idx} the {batch_cnt[idx] + 1} batch")
                            # print(f"idx {idx}  batch size is {batch[0].shape}   {batch[1].shape}")     # [32, 96, 7]  [32, 720, 7]
                            # print('data shape is: ',data.shape)

                            b, t, n = data.shape
                            data = torch.reshape(data.permute(2, 0, 1),(b*n,t,1))   # n, b , t > 


                            b, t, n = data.shape    
                            mask = torch.rand((b, t, n)).to(self.device)
                            mask[mask < self.args.mask_rate] = 0  # masked
                            mask[mask >= self.args.mask_rate] = 1  # remained
                            model_input = data.masked_fill(mask == 0, 0)

                            model_input = data
                            
                            loss = self.model(instruct, model_input, mask=mask, target=model_input)
                            loss = loss / self.gradient_accumulate_every
                            loss.backward()
                            total_loss += loss.item()
                            batch_cnt[idx] += 1
                        except StopIteration:
                            # print('sum is : ', sum(batch_cnt) )
                            # print('')
                            # print(f"Epoch {e+1}: Exiting due to StopIteration.")                           
                            continue
                            # exhausted = True  # 设置标志，跳出 epoch 循环
                            # with torch.no_grad():
                            #     # if self.step != 0 and self.step % self.save_cycle == 0:
                            #     # if self.step != 0 and self.step % self.max_train_batches == 0: 
                            #     if exhausted == True:    
                            #         self.milestone += 1
                            #         if self.milestone % 20 == 0: 
                            #             self.save(self.milestone)
                            #         # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                            #     if self.logger is not None and self.step % self.log_frequency == 0:
                            #         # info = '{}: train'.format(self.args.name)
                            #         # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                            #         # info += ' ||'
                            #         # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                            #         # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                            #         # info += ' | Total Loss: {:.6f}'.format(total_loss)
                            #         # self.logger.log_info(info)
                            #         self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)                               
                            # break
                    # if exhausted:
                    #     continue  # 继续到下一个 epoch        
                    pbar.set_description(f'Step: {step + 1}, Loss: {total_loss:.6f}')

                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sch.step(total_loss)
                    self.opt.zero_grad()
                    self.step += 1
                    step += 1
                    self.ema.update()    
                    # pbar.set_description(f'Step: {step + 1}, Loss: {total_loss:.6f}')
                    # print(f'Step: {step + 1}, Loss: {total_loss:.6f}')

                    with torch.no_grad():
                        # if self.step != 0 and self.step % self.save_cycle == 0:
                        # if self.step != 0 and self.step % self.max_train_batches == 0: 

                        # if self.step != 0 and self.step % (math.ceil(self.train_batches/2)-1) == 0:    
                        #     self.milestone += 1
                        #     if self.milestone % 10 == 0: 
                        #         self.save(self.milestone)
                        #     # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

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
                    if sum(batch_cnt) >= self.train_batches:
                        with torch.no_grad():
                            # if self.step != 0 and self.step % self.save_cycle == 0:
                            # if self.step != 0 and self.step % self.max_train_batches == 0: 
                            
                            self.milestone += 1
                            if self.milestone % 10 == 0: 
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
