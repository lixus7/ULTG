# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## Requirement

Please install the packages in requirements.txt

## Preparation
### Download all the dataset and put them into the folder "data"

## Experiments 

### training and generation
```shell
nohup python -u exe.py --data_name etth_mp --unconditional --nsample 10 --gpu 1 > etth_mp24.log 2>&1 &
nohup python -u exe.py --data_name etth_ep --unconditional --nsample 10 --gpu 0 > etth_ep24.log 2>&1 &
nohup python -u exe.py --data_name energy_mp --unconditional --nsample 10 --gpu 0 > energy_mp24.log 2>&1 &
nohup python -u exe.py --data_name energy_ep --unconditional --nsample 10 --gpu 0 > energy_ep24.log 2>&1 &
nohup python -u exe.py --data_name pems04_mp --unconditional --nsample 10 --gpu 7 > pems04_mp.log 2>&1 &
nohup python -u exe.py --data_name pems04_ep --unconditional --nsample 10 --gpu 7 > pems04_ep.log 2>&1 &
nohup python -u exe.py --data_name pems08_mp --unconditional --nsample 10 --gpu 7 > pems08_mp.log 2>&1 &
nohup python -u exe.py --data_name pems08_ep --unconditional --nsample 10 --gpu 7 > pems08_ep.log 2>&1 &
```


## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
