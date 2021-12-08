# CentripetalText
Codebase for NeurIPS2021 "CentripetalText: An Efficient Text Instance Representation for Scene Text Detection" [[Paper link]](https://arxiv.org/abs/2107.05945)

## Recommended environment
```
Python 3.6+
Pytorch 1.1.0
torchvision 0.3
mmcv 0.2.12
editdistance
Polygon3
pyclipper
opencv-python 3.4.2.17
Cython
numpy
json
```

## Dataset
```none
CentripetalText
└── data
    ├── total_text
    │   ├── Images
    │   │   ├── Train
    │   │   └── Test
    │   └── Groundtruth
    │       └── Polygon
    │           ├── Train
    │           └── Test
    ├── MSRA-TD500
    │   ├── train
    │   └── test
    ├── HUST-TR400
    └── SynthText
        ├── 1
        ├── ...
        ├── 200
        └── gt.mat
```

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/ct/ct_r18_tt.py
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell script
python test.py config/ct/ct_r18_tt.py checkpoints/ct_r18_tt/checkpoint.pth.tar
```

## Evaluation
```shell script
cd eval/
./eval_tt.sh
```

## TODO

- [x] Release code
- [ ] Trained models
- [ ] Recognition codes

## Citation
```
@inproceedings{sheng2021centripetaltext,
    title={CentripetalText: An Efficient Text Instance Representation for Scene Text Detection},
    author={Tao Sheng and Jie Chen and Zhouhui Lian},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021}
}
```