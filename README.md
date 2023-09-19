# RobVolMinNet

## Deep Learning from Noisy Labels via Robust Nonnegative Matrix Factorization-Based Design

Full paper is available [here](IEEE_CAMSAP2023_noisy_label.pdf).

Daniel Wolnick, Shahana Ibrahim, Tim Marrinan, Xiao Fu

## PyTorch Implementation

### Dependencies

we implement our methods with PyTorch on a CUDA GPU cluster.

- [PyTorch](https://PyTorch.org/), version >= 1.7.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 11.1

### Install PyTorch and Torchvision (pip3):
```
pip3 install torch torchvision
```
### Experiments

We verify the effectiveness of RobVolMinNet on several synthetically noisy datasets (MNIST, FashionMNIST, CIFAR10, CIFAR100). Publically available datasets cna be downloaded [here](https://drive.google.com/drive/folders/1OYsRH9x37LQhbmGNv-1Ao1iYTHQN8W7F?usp=sharing) (the images and labels have been processed to .npy format).


### To run the code:
```
python3 main.py --dataset cifar10  --noise_rate 0.50 --percent_instance_noise 0.10
```

additional parameters can be changed using command line parameters.
