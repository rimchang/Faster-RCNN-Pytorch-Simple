# Faster RCNN-Pytorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1506.0149-brightgreen.svg)](https://arxiv.org/abs/1506.01497)


## Pytorch implementation of Faster RCNN.

This is a Pytorch implementation of the paper "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks". I reference [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch) and faster rcnn pytorch tuturial [pytorch pull request](https://github.com/pytorch/examples/pull/21/files) but this request is not completed and closed.

I was about to study faster rcnn code but codes are too difficult to me. I need some simple tutorial but there is no simple code. so I tried to write simple code than other repo.  

this repo's code are 

### Requirements

* pytoch
* tensorflow (Using tensorboad)
* matplotlib


### Usage

I use [floydhub](https://www.floydhub.com/) to train model   
Floydhub is simple deeplearining training tool  

```
pip install -U floyd-cli
```

```
#./input
floyd data init voc
floyd data upload
```
```
#./FRCNN
floyd init frcnn
floyd data status
floyd run --env pytorch --gpu --data [your data id] "python3 main.py"
```

This porject structure is fitted with floydhub structure, so parent directory contain input, output, FRCNN directory  

but you can traning any environment without floydhub

### Training on Pascal VOC 2007

Follow [this project (TFFRCNN)](https://github.com/CharlesShang/TFFRCNN)
to download and prepare the training, validation, test data 
and the VGG16 model pre-trained on ImageNet. 

Since the program loading the data in `FRCNN/input` by default,
you can set the data path as following.

# this repo is not completed. it's performance is low than other rep

### Result
1000 epochs


### Reference


[faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)   
[pytorch pull request](https://github.com/pytorch/examples/pull/21/files).

