# Deep Residual Network For MXNet
a Deep Residual Network Example for MXNet on cifar10 dataset

Paper: [Deep Residual Learning for Image Recognition on arxiv.org](http://arxiv.org/abs/1512.03385)

It has been merge to [MXNet](https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol_resnet-28-small.py)

### Notes:
1. The example has serveral differences to the paper, This example just proves the rule: **The Deeper, The Better**
2. You are welcome to discuss you point of view, for example, You think the batch_normalization should apply in different places, for the author does not say it very clearly

#### Commands & Setups:
- in example/image-classification/train_model.py 
  - set momentum = 0.9, wd = 0.0001, initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2.0)
- in the get_symbol function in example/image-classification/symbol_resnet-28-small.py
  - set n=3(3 for 20 layers, n=9 for 56 layers)
- first train the network with lr=0.1 for 80 epochs
```
python example/image-classification/train_cifar10.py --network resnet-28-small --num-examples 50000 --lr 0.1 --num-epochs 80 --model-prefix cifar10/resnet --batch-size 128
```
- second train the network with lr=0.01 from epoch 81 to epoch 120, with lr=0.001 from epoch 121 to epoch 160
```
python example/image-classification/train_cifar10.py --network resnet-28-small --num-examples 50000 --model-prefix cifar10/resnet --load-epoch 80 --lr 0.01 --lr-factor 0.1 --lr-factor-epoch 40 --num-epochs 200 --batch-size 128
```
in the paper, he train cifar10 for 160 epoch, I set num-epoch to 200 because I want to see whether it is usefull when set lr=0.0001

since it needs 160 epochs, please be patient. And I train with batch size of 128, and train all the models on 1 GPU

#### Test Accuracy:
- for 20 layers resnet, accuracy=0.905+, 0.9125 in the paper
- for 32 layers resnet, accuracy=0.908+, 0.9239 in the paper
- for 56 layers resnet, accuracy=0.915+, 0.9303 in the paper

Though the numbers are a little bit lower than the paper, but it does obey the rule: 
> **the deeper, the better**

#### Differences to the paper on cifar10 network setup
1. in the paper, the author use identity shortcut when dealing with increasing dimensions, while I use 1x1 convolutions to deal with it
2. in the paper, 4 pixels are padded on each side and a 32x32 crop is randomly sampled from the padded image, while I use the dataset provided by mxnet, so the input is 28x28, as a results for 3 different kinds of 2n layers output map sizes are 28x28, 14x14, 7x7, instead of 32x32, 16x16, 8x8 in the paper.

the above two reason might answer why the accuracy is a bit lower than the paper, I suppose.
Off course, there might be other reasons (for example the true network architecture may be different from my script, since my script is just my understanding of the paper), if you find out, please tell me, 


#### Contact information:
- Email: declanxu@gmail.com or declanxu@126.com
- Name: Binbin Xu
- School: Zhejiang University
- Major: Computer Science & Technology
- Degree: Pursuing a Master degree, expected to be graduate at 2017. March

##Thanks
