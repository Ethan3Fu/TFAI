<<<<<<< HEAD
hellow
=======
# TFAI

Implementation of the paper "**A novel self-supervised representation learning framework based on time-frequency alignment and interaction for mechanical fault diagnosis**" by Daxing Fu, Jie Liu, Hao Zhong, Xin Zhang, Fan Zhang.

## Requirements

please install the following packages

* torch
* numpy
* ignite

## Download datasets

[Dataset 1: Axial flow pump dataset](https://pan.baidu.com/s/1MlX03iDfVdmBSMxgN9jhZw?pwd=erpp)

[Dataset 2: XJTUGearbox dataset](https://pan.baidu.com/s/1Pfbbkq0zC3h5pZX5ae2SzA?pwd=gp04)

please download the datasets and place them in the dataset folder:

```
TFAI/
|
└─ dataset/
    |  
    ├─ dataset1  
    ├─ dataset1  
    └─ dataloader.py
```

## Usage

git clone https://github.com/Ethan3Fu/TFAI.git

cd TFAI

### Demo of 100% fine-tuning in Case 1

```
python main.py --mode --test 
```

### Demo of 10% fine-tuning in Case 1

```
python main.py --mode --test --percent 0.1
```

### Demo of 25% fine-tuning in Case 1

```
python main.py --mode --test --percent 0.25
```

### Demo of 50% fine-tuning in Case 1

```
python main.py --mode --test --percent 0.5
```
>>>>>>> 670af4f (upload)
