# NCF: Neural Collaborative Filtering(WWW, 2017)

## Dataset
I used MovieLens dataset(file size: 100K). If there is interaction between user and item, then target value will be 1. so if there is rating value between user and movie, then target value is 1, otherwise 0. for negative sampling, ratio between positive feedback and negative feedback is 1:4 in trainset, and 1:99 in testset. (these ratios are same as author's code @hexiangnan)

## Neural Collaborative Filtering model directory tree
```bash
.
├── README.md
├── main.py
├── data_utils.py
├── GMF.py
├── MLP.py
├── NeuMF.py
├── evaluate.py
└── dataset
    ├── ml-latest-small
    │   └── ratings.csv
    └── ml-latest-small.zip
``` 
## Neural Collaborative Filtering Result
| **MovieLens 100K** |HR|NDCG| Runtime | epoch |learning rate|batchsize|predictive factor| the number of layer |
|:------------------:|:--:|:---:|:-------:|:-----:|:---:|:---:|:---:|:-------------------:|
|        GMF         |0.825|0.554|   2m    |  20   |0.001|256|8|          X          |
|        MLP         |0.825|0.559|  2.8m   |  20   |0.001|256|8|          3          |
|NeuMF(without pre-training)|0.848|0.578|   3m    |  20   |0.001|256|8|          3          |
| NeuMF(with pre-training)  |0.875|0.577|  2.6m   |  20   |0.001|256|8|          X          |


### Development Enviroment
- OS: Max OS X
- IDE: pycharm
- GPU: NVIDIA RTX A6000

### Quick Start Example
```bash
python main.py -m GMF -nf 8 -b 256 -e 20 -lr 0.001 -tk 10 -pr False -save True
``` 
```bash
python main.py -m MLP -nf 8 -nl 3 -b 256 -e 20 -lr 0.001 -tk 10 -pr False -save True
```  
```bash
python main.py -m NeuMF -nf 8 -nl 3 -b 256 -e 20 -lr 0.001 -tk 10 -pr False
```  
```bash
python main.py -m NeuMF -nf 8 -nl 3 -b 256 -e 20 -lr 0.001 -tk 10 -pr True
```  
### Reference
paper : [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

review written in korean : [Review](https://ikgyu-lee.notion.site/WWW-2017-Neural-Collaborative-Filtering-a9b9f9dee46a4c289536570ddd08e5f8)

Neural Collaborative Filtering with MovieLens in torch

In progress 'with pre-training'