# Diffusion-TS: Interpretable Diffusion for General Time Series Generation

[![](https://img.shields.io/github/stars/Y-debug-sys/Diffusion-TS.svg)](https://github.com/Y-debug-sys/Diffusion-TS/stargazers)
[![](https://img.shields.io/github/forks/Y-debug-sys/Diffusion-TS.svg)](https://github.com/Y-debug-sys/Diffusion-TS/network) 
[![](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/LICENSE) 
<img src="https://img.shields.io/badge/python-3.8-blue">
<img src="https://img.shields.io/badge/pytorch-2.0-orange">

> **Abstract:** Denoising diffusion probabilistic models (DDPMs) are becoming the leading paradigm for generative models. It has recently shown breakthroughs in audio synthesis, time series imputation and forecasting. In this paper, we propose Diffusion-TS, a novel diffusion-based framework that generates multivariate time series samples of high quality by using an encoder-decoder transformer with disentangled temporal representations, in which the decomposition technique guides Diffusion-TS to capture the semantic meaning of time series while transformers mine detailed sequential information from the noisy model input. Different from existing diffusion-based approaches, we train the model to directly reconstruct the sample instead of the noise in each diffusion step, combining a Fourier-based loss term. Diffusion-TS is expected to generate time series satisfying both interpretablity and realness. In addition, it is shown that the proposed Diffusion-TS can be easily extended to conditional generation tasks, such as forecasting and imputation, without any model changes. This also motivates us to further explore the performance of Diffusion-TS under irregular settings. Finally, through qualitative and quantitative experiments, results show that Diffusion-TS achieves the state-of-the-art results on various realistic analyses of time series. 

Diffusion-TS is a diffusion-based framework that generates general time series samples both conditionally and unconditionally. As shown in Figure 1, the framework contains two parts: a sequence encoder and an interpretable decoder which decomposes the time series into seasonal part and trend part. The trend part contains the polynomial regressor and extracted mean of each block output. For seasonal part, we reuse trigonometric representations based on Fourier series. Regarding training, sampling and more details, please refer to [our paper](https://openreview.net/pdf?id=4h1apFjO99) in ICLR 2024. 

<p align="center">
  <img src="figures/fig1.jpg" alt="">
  <br>
  <b>Figure 1</b>: Overall Architecture of Diffusion-TS.
</p>


## Dataset Preparation

All the four real-world datasets (Stocks, ETTh1, Energy and fMRI) can be obtained from [Google Drive](https://drive.google.com/file/d/11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6/view?usp=sharing). Please download **dataset.zip**, then unzip and copy it to the folder `./Data` in our repository.


## Running the Code

 The code requires conda3 (or miniconda3), and one CUDA capable GPU. The instructions below guide you regarding running the codes in this repository. 

### Environment & Libraries

The full libraries list is provided as a `requirements.txt` in this repo. Please create a virtual environment with `conda` or `venv` and run

~~~bash
(myenv) $ pip install -r requirements.txt
~~~

### Training & Sampling

For training, you can reproduce the experimental results of all benchmarks by runing

```bash
nohup python -u main.py --gpu 5 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_etth evening_peak_etth morning_peak_energy  evening_peak_energy morning_peak_pems04  evening_peak_pems04 morning_peak_pems08 evening_peak_pems08 --sample 0 --train --epoch 1000 --batch 8 > mix_nogpt_train_etthmpep_energympep_pems04mpep_pems08mpep_1000_mask0.0.log 2>&1  && (nohup python -u main.py --gpu 2 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_etth --sample 0 --milestone 1000 > test_etthmp_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 3 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_etth --sample 0 --milestone 1000 > test_etthep_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 7 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_energy --sample 0 --milestone 1000 > test_energymp_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 1 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_energy --sample 0 --milestone 1000 > test_energyep_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 2 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_pems04 --sample 0 --milestone 1000 > test_pems04mp_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 3 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_pems04 --sample 0 --milestone 1000 > test_pems04ep_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 7 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_pems08 --sample 0 --milestone 1000 > test_pems08mp_nogptmix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 1 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_pems08 --sample 0 --milestone 1000 > test_pems08ep_nogptmix_2layer_mile1000_mask0.0.log 2>&1 &) &
```


## Visualization and Evaluation

After sampling, synthetic data and orginal data are stored in `.npy` file format under the *output* folder, which can be directly read to calculate quantitative metrics such as discriminative, predictive, correlational and context-FID score. You can also reproduce the visualization results using t-SNE or kernel plotting, and all of these evaluational codes can be found in the folder `./Utils`. Please refer to `.ipynb` tutorial files in this repo for more detailed implementations.

**Note:** All the metrics can be found in the `./Experiments` folder. Additionally, by default, for datasets other than the Sine dataset (because it do not need normalization), their normalized forms are saved in `{...}_norm_truth.npy`. Therefore, when you run the Jupternotebook for dataset other than Sine, just uncomment and rewrite the corresponding code written at the beginning.

### Main Results

#### Standard TS Generation
<p align="center">
  <b>Table 1</b>: Results of 24-length Time-series Generation.
  <br>
  <img src="figures/fig2.jpg" alt="">
</p>

#### Long-term TS Generation
<p align="center">
  <b>Table 2</b>: Results of Long-term Time-series Generation.
  <br>
  <img src="figures/fig3.jpg" alt="">
</p>

#### Conditional TS Generation
<p align="center">
  <img src="figures/fig4.jpg" alt="">
  <br>
  <b>Figure 2</b>: Visualizations of Time-series Imputation and Forecasting.
</p>


## Authors

* Paper Authors : Xinyu Yuan, Yan Qiao

* Code Author : Xinyu Yuan

* Contact : yxy5315@gmail.com


## Citation
If you find this repo useful, please cite our paper via
```bibtex
@inproceedings{yuan2024diffusionts,
  title={Diffusion-{TS}: Interpretable Diffusion for General Time Series Generation},
  author={Xinyu Yuan and Yan Qiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=4h1apFjO99}
}
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/cientgu/VQ-Diffusion

https://github.com/XiangLi1999/Diffusion-LM

https://github.com/philipperemy/n-beats

https://github.com/salesforce/ETSformer

https://github.com/ermongroup/CSDI

https://github.com/jsyoon0823/TimeGAN
