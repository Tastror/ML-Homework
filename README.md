# ML Homework

work 1: handwriting detect

## Train

```shell
python train.py [--nred] [--naug]
```

`--nred`: no redundant, recommed to open  
`--naug`: no augmentation, do not recommed to open  
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
