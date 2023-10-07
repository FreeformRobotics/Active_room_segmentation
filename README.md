# Active room segmentation

This is the Code for the paper 'Human Cognition-Inspired Active Room Segmentation'. Inspired by the human cognition system, this method incorporates vision input as an additional feature and follows a room-by-room exploration strategy to facilitate both the room exploration and exploration tasks. For full details refer to [the paper]().

# Dependencies
## Installing habitat

The habitat-sim and habitat-api used in this method are the  same as [ANS](https://github.com/devendrachaplot/Neural-SLAM). Please refer to [ANS](https://github.com/devendrachaplot/Neural-SLAM) for installing the specific version of habitat-sim and habitat-api.
## Setup
After installing habitat, clone the repository and install other requirements:
```
git clone https://github.com/B0GGY/Active_room_segmentation.git
cd Active_room_segmentation
pip install -r requirements.txt
```
## Dateset
To download the Gibson scene dataset and task datasets(Point goal navigation), please refer to [this site](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). 

# Usage
For running the active room segmentation method:
```
python explorable_with_door_detection.py --split val --eval 1 -n 1 -v 1 --train_global 0 --train_local 0 --train_slam 0 
```

