# Active room segmentation

This is the Code for the paper 'Human Cognition-Inspired Active Room Segmentation'. Inspired by the human cognition system, this method incorporates vision input as an additional feature and follows a room-by-room exploration strategy to facilitate both the room exploration and exploration tasks. For full details refer to [the paper]().

# Dependencies
## Installing habitat

The habitat-sim and habitat-api used in this method are the  same as ANS. Please refer to [ANS](https://github.com/devendrachaplot/Neural-SLAM) for installing the specific version of habitat-sim and habitat-api.
## Setup
After installing habitat, clone the repository and install other requirements:
```
git clone https://github.com/B0GGY/Active_room_segmentation.git
cd Active_room_segmentation
pip install -r requirements.txt
```
## Door detection network
In this method, we borrow the door detection network from [aislabunimi](https://github.com/aislabunimi/door-detection-long-term). The train params of the network can be downloaded from [here](https://unimi2013-my.sharepoint.com/personal/michele_antonazzi_unimi_it/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fmichele%5Fantonazzi%5Funimi%5Fit%2FDocuments%2Ftrain%5Fparams%2Fdetr%5Fresnet50%5F4). After downloading and unzipping it:
```
cd detr_door_detection
mkdir -p train_params/detr_resnet_50_4
mv 'location of the downloaded final_doors_dataset' train_params/detr_resnet_50_4
```

## Dateset
To download the Gibson scene dataset and task datasets(Point goal navigation), please refer to [this site](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). 

# Usage
For running the active room segmentation method:
```
python explorable_with_door_detection.py --split val --eval 1 -n 1 -v 1 --train_global 0 --train_local 0 --train_slam 0 
```

