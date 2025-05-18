import os
import torch
import argparse
import numpy as np
import json
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', nargs='+', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--instruct_path', type=str, default='./Data/datasets/prompts.json', 
                        help='prompt')    
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mask_rate', type=float, default=0.,  help='mask rate.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('--current_ins', type=str, default=' ', 
                        help='ins for inference')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    configfile = args.config_file 
    print('all config files: ',configfile)
    f = open(args.instruct_path)
#     results = {key: instruct_list[key] for key in args.config_file if key in instruct_list}
#     print('Avalable instruct_list: ',results) 

    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)    
    
    loaders = []
    train_batches = 0
    ett_ins = {'morning_peak_etth':'ETTMP', 'evening_peak_etth':'ETTEP', 'morning_peak_energy':'ENEMP', 'evening_peak_energy':'ENEEP','morning_peak_pems04':'PEMS04MP', 'evening_peak_pems04':'PEMS04EP', 'morning_peak_pems08':'PEMS08MP', 'evening_peak_pems08':'PEMS08EP','workdays': 'WORK', 'weekends': 'WEEKEND','daytime': 'DAT','nighttime': 'NIGHT','etth': 'etth','energy': 'energy',} # , 'WORK', 'WEEKEND', 'DAY', 'NIGHT'
    
    ins=[]
    max_train_batches = 0

    for config_item in configfile:
        ins.append(ett_ins[config_item])
        # ins.append(config_item)
        config_path = "./Config/" + config_item + ".yaml"
        config = load_yaml_config(config_path)
        print('config file is', config)
        config = merge_opts_to_config(config, args.opts)

        logger = Logger(args)
        logger.save_config(config)
        dataloader = build_dataloader(config, args)['dataloader']
        dataloader_info = build_dataloader(config, args)
        loaders.append(dataloader)
        # print('len(dataloader_info) SIZE :' ,len(dataloader_info))
        train_batches += len(dataloader)
        max_train_batches = max(len(dataloader), max_train_batches)
        # print('dataset : ',dataloader_info['dataset'])
        # print('dataloader  : ',dataloader_info['dataloader'])
    model = instantiate_from_config(config['model']).cuda()
    print('instruct_list: ',ins)


    
    # for config_item in configfile:
    #     ins.append(ett_ins[config_item])
    #     # ins.append(config_item)
    #     config_path = "./Config/" + config_item + ".yaml"
    #     config = load_yaml_config(config_path)
    #     print('config file is', config)
    #     config = merge_opts_to_config(config, args.opts)

    #     logger = Logger(args)
    #     logger.save_config(config)
    #     dataloader = build_dataloader(config, args)['dataloader']
    #     dataloader_info = build_dataloader(config, args)
    #     loaders.append(dataloader)
    #     # print('len(dataloader_info) SIZE :' ,len(dataloader_info))
    #     print('len(dataloader) SIZE :' ,len(dataloader))
    #     train_batches += len(dataloader)
    #     max_train_batches = max(len(dataloader), max_train_batches)
    #     # print('dataset : ',dataloader_info['dataset'])
    #     # print('dataloader  : ',dataloader_info['dataloader'])
    # model = instantiate_from_config(config['model']).cuda()
    # print('instruct_list: ',ins)
        
    print('train batches : ', train_batches)
    trainer = Trainer(config=config, args=args, model=model, ins = ins, dataloader=loaders, train_batches=train_batches, max_train_batches=max_train_batches, logger=logger)

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(instruct=ins[0] ,num=len(dataset)* dataset.var_num, size_every=900, shape=[dataset.window, 1])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.config_file[0]}_milestone_{args.milestone}.npy'), samples)

if __name__ == '__main__':
    main()
