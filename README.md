# ML Homework

work 1: handwriting detect

**More in [readme.ipynb (Chinese)](./readme.ipynb)**  
**Dataset characters' name in [dataset/NewDataset.txt (Chinese)](./dataset/NewDataset.txt)**

## Train

```shell
python train.py [--nred] [--naug]
```

`--nred`: no redundant, recommend to use  
`--naug`: no augmentation, not recommend to use  
result will save in `weight/xxx.pt`

if use SVM, just use `python train_svm.py --naug --nred` (it will be pretty slow if not add `--naug` (means use augmentation))

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

![inference gif](https://s2.loli.net/2023/03/31/FwOg6JqhIX4aZuB.gif)

## Show

```shell
python show_model.py
python show_model.py pt-torch-123456.pt
```

directory name is `weight/`  
default use `pt-torch-best.pt`

![show network png](https://s2.loli.net/2023/04/03/9ukLUo4WYl2Kqng.png)
