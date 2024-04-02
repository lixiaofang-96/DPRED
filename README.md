# DPRED
Code used for "**Deep parametric Retinex decomposition model for low-light image enhancement**" (Computer Vision and Image Understanding)

## Network
![DPRED](https://github.com/lixiaofang-96/DPRED/blob/main/network.png)

## Implementation
```
Pytorch 1.10.1
Python 3.6.12
```

## Train
```
python DecomNet_train.py
python Enhance_I_train.py
python Enhance_R_train.py
python Denoise_enI_train.py
```

## Test
```
python test_LOL.py
python test_LIME.py
```

## Model
model.py

## Load Dataset
MyDataset.py

## Citation
if you find this repo is helpful, please cite

```
@article{li2024deep,
  title={Deep parametric Retinex decomposition model for low-light image enhancement},
  author={Li, Xiaofang and Wang, Weiwei and Feng, Xiangchu and Li, Min},
  journal={Computer Vision and Image Understanding},
  volume={241},
  pages={103948},
  year={2024}
}
```
