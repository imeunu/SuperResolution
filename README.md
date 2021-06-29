# Super Resolution Imaging with Deep Neural Networks

Single Image Super Resolution(SISR) is a class of techniques that enhance the resolution of an imaging system.
Goal of SISR is converting Low Resolution(LR) image into High Resolution(HR) image. Super Resolution requires a high-resolution restore of low-resolution images, which may have multiple correct answers, which are targets to restore. In other words, it means an undefined problem in which the only correct answer does not exist, which is called Ill-posed problem.

First, set HR image as a Ground Truth(GT) target, and apply blurring or down sampling to make LR image. It then follows the structure of training the model to restore the low resolution image to GT through some method. This is the fundamental limitation of the Single Image Super Resolution, and the performance of the Super Resolution may vary depending on the distortion, down sapling technique used to create low-resolution images.

## [SRCNN](https://github.com/imeunu/SuperResolution/tree/main/SRCNN)
## [VDSR](https://github.com/imeunu/SuperResolution/tree/main/VDSR)
## [SRGAN](https://github.com/imeunu/SuperResolution/tree/main/SRGAN)
