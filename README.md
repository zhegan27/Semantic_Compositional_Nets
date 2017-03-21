# Semantic Compositional Networks

The Theano code for the CVPR 2017 paper “[Semantic Compositional Networks for Visual Captioning](https://arxiv.org/pdf/1611.08002.pdf)”

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7 (do not use Python 3.0)
* Theano 0.7 (you can also use the most recent version)
* A recent version of NumPy and SciPy 

## Getting started

We provide the code on how to train SCN for image captioning on the COCO dataset. 

* In order to start, please first download the [ResNet features and tag features](https://drive.google.com/open?id=0B1HR6m3IZSO_QmZVV3hTbmJwRFU) we used in the experiments. Put the  `coco` folder inside the `./data` folder.

* We also provide our [pre-trained model](https://drive.google.com/open?id=0B1HR6m3IZSO_QmZVV3hTbmJwRFU) on COCO. Put the `pretrained_model` folder into the current directory.

* In order to evaluate the model, please download the standard [coco-caption evaluation code](https://github.com/tylin/coco-caption). Copy the folder `pycocoevalcap` into the current directory.

* Now, everything is ready.

## How to use the code

1. Run `SCN_training.py` to start training. On a modern GPU, the model will take one night  to train.

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python SCN_training.py 
```

2. Based on our pre-trained model, run `SCN_decode.py` to generate captions on the COCO small 5k test set. The generated captions are also provided, named `coco_scn_5k_test.txt`.

3. Now, run `SCN_evaluation.py` to evaluate the model. The code will output

```
CIDEr: 1.043, Bleu-4: 0.341, Bleu-3: 0.446, Bleu-2: 0.582, Bleu-1: 0.743, ROUGE_L: 0.550, METEOR: 0.261. 
```

4. In the `./data/coco` folder, we also provide the features for the COCO official validation and test sets. Run `SCN_for_test_server.py` will help you generate captions for the official test set, and prepare the `.json` file for submission. 

Model architecture and illustration of semantic composition.
<img src="figure1.png" width="400px">

## Citing SCN

Please cite our CVPR paper in your publications if it helps your research:

    @inproceedings{SCN_CVPR2017,
      Author = {Gan, Zhe and Gan, Chuang and He, Xiaodong and Pu, Yunchen and Tran, Kenneth and Gao, Jianfeng and Carin, Lawrence and Deng, Li},
      Title = {Semantic Compositional Networks for Visual Captioning},
      booktitle={CVPR},
      Year  = {2017}
    }

 