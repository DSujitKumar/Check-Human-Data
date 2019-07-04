# Check Human Data
This is a python using Facenet and their pretrained model.
Mostly it's Work to check the presence of a person in the file or not using image.
## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.
## Inspiration
This repository is heavily inspired by David Sandberg (https://github.com/davidsandberg) Who made the Facenet Model. and Arun Mandal(https://github.com/arunmandal53) of making the code usability for me.

## Running Program
#### pip install requirement.txt
#### python Check_Existance.py --img1=Katrina1.jpg
