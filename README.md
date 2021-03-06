## MAT-net:representing appearance-irrelevant warp field by multiple affine transformations
Pytorch implementation of "MAT-net:representing appearance-irrelevant warp field by multiple affine transformations"

### prerequisites
* pytorch==1.0.0
* the version of pytorch is important. Other packages should be installed by following the output of shell.
### Datasets
* Voxceleb: following https://github.com/DaoiestFire/video-preprocessing
* FashionVideo: following https://vision.cs.ubc.ca/datasets/fashion/
* moving-Gif: download from [google drive](https://drive.google.com/drive/folders/1TxjQFzeNk4CSlvP92xbmGt20zBzJJUw9)
* the structure of dataset like:
```
dataset_folder-
              |--train_folder-
                             |--train_sample1
                             |--train_sample2
                             |...
              |--test_folder-
                             |--test_sample1
                             |--test_sample2
                             |...
```
### pretrained-model
download from [google drive](https://drive.google.com/drive/folders/1mEF07YEuY4bOklM3m9tBvuzDPRpLMR7s)
### Demo
* Note that `relative animation` is recommended. i.e. an init image is needed. `init image` is similar to `input image` in pose.
* animate the input image according to one target image
```bash
python demo.py --config path/to/config --checkpoint path/to/checkpoint --source path/to/input_image --driving path/to/driving_image --result path/to/save/result --image --init /path/to/init
```
`--cpu` can be added to use cpu device.
* animate the input image according to one target video

for face data, `--find_best_frame` means to find `init image` automaticly.
```bash
python demo.py --config path/to/config --checkpoint path/to/checkpoint --source path/to/input_image --driving path/to/driving_video --result path/to/save/result --find_best_frame
```
you also can speicfy a frame of driving video as the init image
```bash
python demo.py --config path/to/config --checkpoint path/to/checkpoint --source path/to/input_image --driving path/to/driving_video --result path/to/save/result --best_frame best_frame_number
```
if command without `--find_best_frame` and `--best_frame`, demo will use the first frame of driving video as `init image`. or you can specify `--relative False` to use absolute transfer.
### Train
to train on a dataset
```bash
python run.py --config path/to/config
```
to train from a checkpoint
```bash
python run.py --config path/to/config --checkpoint path/to/checkpoint
```
the output of training process will be saved in `./log` folder.
### Test reconstruction
`reconstruction` means reconstruct every videos in test folder of dataset.
```bash
python run.py --config path/to/config --checkpoint path/to/checkpoint --mode reconstruction
```
resulys will be saved in `./log/reconstruction`
### Test animation
`animation` means animate a still image according to a target video. video pairs has been saved in `config/dataset_name.csv`
```bash
python run.py --config path/to/config --checkpoint path/to/checkpoint --mode animate
```

### update
2021/01/26 pretrained models have been released.

2021/01/24 Test has been updated.

2021/01/23 Datasets, Train has been updated.

2021/01/21 Demo has been updated.
### Acknowledgement
our project is based on [first order motion model](https://github.com/AliaksandrSiarohin/first-order-model)
