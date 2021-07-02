# Super Resolution Imaging with Deep Neural Networks

Single Image Super Resolution(SISR) is a class of techniques that enhance the resolution of an imaging system.
Goal of SISR is converting Low Resolution(LR) image into High Resolution(HR) image. Super Resolution requires a high-resolution restore of low-resolution images, which may have multiple correct answers, which are targets to restore. In other words, it means an undefined problem in which the only correct answer does not exist, which is called Ill-posed problem. Set HR image as a Ground Truth(GT) target, and apply blurring or down sampling to make LR image. It then follows the structure of training the model to restore the low resolution image to GT through some method. This time, I will implement 3 Deep Neural Network based methods.

## [SRCNN](https://github.com/imeunu/SuperResolution/tree/main/SRCNN)
To implement [Image Super-Resolution Using Deep Convolutional Networks(ECCV 2014).](https://arxiv.org/abs/1501.00092),

## [VDSR](https://github.com/imeunu/SuperResolution/tree/main/VDSR)
## [SRGAN](https://github.com/imeunu/SuperResolution/tree/main/SRGAN)
