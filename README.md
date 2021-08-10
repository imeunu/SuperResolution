# Super Resolution Imaging with Deep Neural Networks

Single Image Super Resolution(SISR) is a class of techniques that enhance the resolution of an imaging system.
Goal of SISR is converting Low Resolution(LR) image into High Resolution(HR) image. Super Resolution requires a high-resolution restore of low-resolution images, which may have multiple correct answers, which are targets to restore. In other words, it means an undefined problem in which the only correct answer does not exist, which is called Ill-posed problem. Set HR image as a Ground Truth(GT) target, and apply blurring or down sampling to make LR image. It then follows the structure of training the model to restore the low resolution image to GT through some method. This time, I will implement 3 Deep Neural Network based methods.

## [SRCNN](https://github.com/imeunu/SuperResolution/tree/main/SRCNN)
[Image Super-Resolution Using Deep Convolutional Networks(ECCV 2014)](https://arxiv.org/abs/1501.00092) reinterpreted sparse-coding based SR method as deep convolutional network. 

**Implement Details**
- 3 Convolutional layers of filter size 9, 1, 5
- Channel Size: 64, 32, 1
- Training Data: [91 images](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
- Image Patch Size: 33 (decomposed into 24,800 sub-images)
- Validation Data: [Set5](https://paperswithcode.com/dataset/set5) was used.

## [VDSR](https://github.com/imeunu/SuperResolution/tree/main/VDSR)
[Accurate Image Super-Resolution Using Very Deep Convolutional Networks(CVPR 2016)](https://arxiv.org/abs/1511.04587) was inspired by VGG-net. It supplemented the limitations of SRCNN with several techniques by utilizing contextual information about large image regions to be exploited.

**Implement Details**
- 20 Convolutional layers with filter size 3, zero padding
- Extremely high learning rates with Adjustable Gradient Clipping
- Residual Learning
- SGD momentum: 0.9
- Regularized by weight decay(L2 penalty multiplied by 0.0001)

## [SRGAN](https://github.com/imeunu/SuperResolution/tree/main/SRGAN)
