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
* timm used (pip)

* VGG 19 is used without batchnorm (it is not declared in the paper which VGG19 is used)
* Swin B (base) is used as the transformer model (it is also not declared) (not swin2, swin1 is used)
* In the loss functions, the paper does not explicitly state if squared error or euclidean distance is used. Despite the cited papers for the loss functions use euclidean distance, we used squared error as the loss values aligned better with the paper this way.
* Again, the paper does not explicitly state if the loss functions are scaled by the Batch Size, Channel Size, Image Height or Image Width. We take the mean of the squared distances, which scales the loss by the batch size and channel size (also by the image height and width in the content loss), as it were more plausible and the loss values aligned better with the paper this way.
* We don't know if we should normalize the dataset images with the mean and std of the dataset. So, we did not do it.
