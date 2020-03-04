# Pytorch Toy Project

## 1. Introduction

- This toy projected is intended to compare the performance of CNNs with Dynamic Context Module (DCM) 
and Adaptive Context Module (ACM) with that of baselines'.  
- For convenience and efficiency, I suggest only using mini or small dataset, say `CIFAR10`, `MNIST` or `FashionMNIST`
, in order to save time and computation costs. (Those datasets can be used at ease with `torchvision` library)
- This project has a lot of deficiencies now, and I'll update it in the future.

## 2. How To Use
1. First, initial setup:

      ``` shell script
       git clone https://github.com/Willzy-x/pytorch-toy-project.git
       mkdir results data
       pip install -r requirements.txt
      ```
 
2. Then change the hyper-parameters in `main.py`, (e.g. `LEARNING_RATE`, `WEIGHT_DECAY` and so on.)

3. Last, type `python main.py` in command line to start experiment.

## 3. Some Features

- Visualize training and testing process with `visdom`.
    - Type `python -m visdom.server` to start visdom server.
    - If you are running visdom server on your computer locally, just type `localhost:8097` in browser.
    - Please find out more in https://github.com/facebookresearch/visdom if necessary.

- Some vanilla CNNs are provided in models folder, like AlexNet, ResNet and VGG. Of course, you can also use those networks
in `torchvision` library.

## 4. References
1. He J, Deng Z, Qiao Y. Dynamic Multi-scale Filters for Semantic Segmentation[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 3562-3572.  
2. He J, Deng Z, Zhou L, et al. Adaptive pyramid context network for semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 7519-7528.

