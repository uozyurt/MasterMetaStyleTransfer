# ceng_796_project_yigit_ozyurt
Master: Meta Style Transformer for Controllable Zero-Shot and Few-Shot Artistic Style Transfer


Private (for now) repository to conduct experiments for the project of CENG 796, spring 2024 course.


This is an re-implementation of the paper (without official code implementation) stated above.



Project owners: Melih Gökay Yiğit, Umut Özyurt




wiki art download link: https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view (25.4 GB)
coco download link: http://images.cocodataset.org/zips/train2017.zip (18 GB)




notes and assumptions:

* python 3.10 is used
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
* pytorch 1.13.1 is used
* torchvision 0.14.1 is used
* pytorch-cuda 11.6 is used


* VGG 19 is used without batchnorm (it is not declared in the paper which VGG19 is used)
* Swin B (base) is used as the transformer model (it is also not declared) (not swin2, swin1 is used)
* in the loss functions, the cited papers scale down the losses with C*W*H, but it is not explicit in the paper. Still, if no scaling is used, losses become too high, so we assumed it should be scaled down.
* 
