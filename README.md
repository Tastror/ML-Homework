# ML Homework

work 1: handwriting detect

**More in [readme.ipynb](./readme.ipynb)**  
**Dataset characters' name in [dataset/NewDataset.txt](./dataset/NewDataset.txt)**

## Train

```shell
python train.py [--nred] [--naug]
```

`--nred`: no redundant, recommend to use  
`--naug`: no augmentation, not recommend to use  
result will save in `weight/xxx.pt`

## Evaluate

```shell
python eval.py [--nred]
python eval.py [--nred] pt-torch-123456.pt
```

directory name is `weight/`  
default use `pt-torch-best.pt` or `pt-torch-nred-best.pt` (depends on whether `--nred` is used)  
`--nred`: no redundant, should be the same as the training method of `xxx.pt`

## Inference

```shell
python infer.py
python infer.py pt-torch-123456.pt
```

directory name is `weight/`  
default use `pt-torch-best.pt`
