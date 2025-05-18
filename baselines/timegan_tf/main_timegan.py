"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示ERROR级别的日志

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
import os

import os
import time
import numpy as np


def main(args):
    """Main function for timeGAN experiments.
    
    Args:
      - data_name: sine, stock, or energy
      - seq_len: sequence length
      - Network parameters (should be optimized for different datasets)
        - module: gru, lstm, or lstmLN
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
      - metric_iteration: number of iterations for metric computation
    
    Returns:
      - ori_data: original data
      - generated_data: generated synthetic data
      - metric_results: discriminative and predictive scores
    """
    # Start timing
    start_time = time.time()

    ## Data loading
    ori_data = real_data_loading(args.data_name, args.seq_len)
    print(f"{args.data_name} dataset is ready.")

    print('ori_data [0] shape is: ', ori_data[0].shape)
    ori_npy = np.array(ori_data)
    print('ori_data save shape is: ',ori_npy.shape)
    # b,t,n = ori_npy.shape
    # ori_npy = ori_npy.transpose(2, 0, 1 ).reshape(b*n,t,1)

    save_dir = os.path.join(args.output_dir, args.data_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ori_{args.data_name}{args.seq_len}.npy")
    np.save(save_path, ori_npy)
    print(f"{args.data_name} dataset is saved.")

    ## Synthetic data generation by TimeGAN
    parameters = {
        'module': args.module,
        'hidden_dim': args.hidden_dim,
        'num_layer': args.num_layer,
        'iterations': args.iteration,
        'batch_size': args.batch_size
    }
    generated_data, gene_np = timegan(ori_data, parameters)
    print('Finish Synthetic Data Generation')

    print('generated_data shape is ', gene_np.shape)
    save_path = os.path.join(save_dir, f"generate_{args.data_name}{args.seq_len}.npy")
    np.save(save_path, gene_np)
    print(f"Finished saving generated data {args.data_name}")

    ## Performance metrics   
    metric_results = {}

    # 1. Discriminative Score
    discriminative_score = []
    for _ in range(args.metric_iteration):
        discriminative_score.append(discriminative_score_metrics(ori_data, generated_data))
    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = []
    for _ in range(args.metric_iteration):
        predictive_score.append(predictive_score_metrics(ori_data, generated_data))
    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, generated_data, 'pca')
    visualization(ori_data, generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)

    # End timing
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")

    return ori_data, generated_data, metric_results



if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      # choices=['sine','stock','energy'],
      default='stock',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument('--output_dir', type=str, default="./output", help='folder to output metrics and images')
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  # parser.add_argument('--gpu', type=int, default=None,
  #                   help='GPU id to use. If given, only the specific gpu will be'
  #                   ' used, and ddp will be disabled')
  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)