## Search and Pruning (SnP) 

This repository includes our code for the paper 'Large-scale Training Data Search for Object Re-identification' in CVPR2023.

Related material: [Paper](https://arxiv.org/abs/1912.08855), [Video](https://youtu.be/OAZ0Pka2mKE)

![fig1](https://github.com/yorkeyao/SnP/blob/main/images/SnP.jpg)  

As shown in figure above, we present a search and pruning (SnP) solution to the training data search problem in object re-ID. The source data pool is 1 order of magnitude larger than existing re-ID training sets in terms of the number of images and the number of identities. When the target is AlicePerson, from the source pool, our method (SnP) results in a training set 80\% smaller than the source pool while achieving a similar or even higher re-ID accuracy. The searched training set is also superior to existing individual training sets such as Market-1501, Duke, and MSMT.

## Requirements

- Sklearn
- Scipy 1.2.1
- PyTorch 1.7.0 + torchivision 0.8.1

## Re-ID Dataset Preparation

![fig1](https://github.com/yorkeyao/SnP/blob/main/images/datasets.jpg)  

Please prepare the following datasets for person re-ID: [DukeMTMC-reID](https://exposing.ai/duke_mtmc/), [Market1503](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html), [MSMT17](http://www.pkuvmc.com/publications/msmt17.html), [CUHK03](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), [RAiD](https://cs-people.bu.edu/dasabir/raid.php), [PersonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset), [UnrealPerson](https://github.com/FlyHighest/UnrealPerson), [RandPerson](https://github.com/VideoObjectSearch/RandPerson), [PKU-Reid](https://github.com/charliememory/PKU-Reid-Dataset), [VIPeR](https://vision.soe.ucsc.edu/node/178), [AlicePerson (target data in VisDA20)](https://github.com/Simon4Yan/VisDA2020).

You may need to sign up to get access to some of these datasets. Please store these datasets in a file strcuture like this

```
~
└───reid_data
    └───duke_reid
    │   │ bounding_box_train
    │   │ ...
    │
    └───market
    │   │ bounding_box_train
    │   │ ...
    │
    └───MSMT
    │   │ MSMT_bounding_box_train
    │   │ ...
    │
    └───cuhk03_release
    │   │ cuhk-03.mat
    │   │ ...
    │
    └───alice-person
    │   │ bounding_box_train
    │   │ ...
    │
    └───RAiD_Dataset-master
    │   │ bounding_box_train
    │   │ ...
    │
    └───unreal
    │   │ UnrealPerson-data
    │   │ ...
    │
    └───randperson_subset
    │   │ randperson_subset
    │   │ ...
    │
    └───PKU-Reid
    │   │ PKUv1a_128x48
    │   │ ...
    │
    └───i-LIDS-VID
    │   │ PKUv1a_128x48
    │   │ ...
    │
    └───VIPeR
    │   │ images
    │   │ ...
```

Please prepare the following datasets for vehicle re-ID: [VeRi](https://github.com/JDAI-CV/VeRidataset), [CityFlow-reID](https://www.aicitychallenge.org/), [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-wild](https://github.com/PKU-IMRE/VERI-Wild), [VehicleX](https://drive.google.com/file/d/1qySICqFJdgjMVi6CrLwVxEOuvgcQgtF_/view?usp=sharing), [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [PKU-vd1 and PKU-vd2](https://www.pkuml.org/resources/pku-vds.html). The AliceVehicle will be public available by our team shortly. 

Please store these datasets in a file strcuture like this

```
~
└───reid_data
    └───VeRi
    │   │ bounding_box_train
    │   │ ...
    │
    └───AIC19-reid
    │   │ bounding_box_train
    │   │ ...
    │
    └───VehicleID_V1.0
    │   │ image
    │   │ ...
    │
    └───vehicleX_random_attributes
    │   │ ...
    │
    └───veri-wild
    │   │ VeRI-Wild
    │   │ ...
    │
    └───stanford_cars
    │   │ PKUv1a_128x48
    │   │ ...
    │
    └───i-LIDS-VID
    │   │ PKUv1a_128x48
    │   │ ...
    │
    └───VIPeR
    │   │ images
    │   │ ...
```

## Running example 

When Market is used as target, we can seach a training set with 2860 IDs using the command below:

```python
python trainingset_search_person.py --target 'market' \
--result_dir 'results/sample_data_market/' --n_num_id 2860 \
--ID_sampling_method SnP --img_sampling_method 'FPS' --img_sampling_ratio 0.5 \
--output_data '/data/reid_data/market/SnP_2860IDs_0.5Imgs_0610'  
```

When VeRi is used as target, the command is:

```python
python trainingset_search_vehicle.py --target 'veri' \
--result_dir './results/sample_data_veri/' --n_num_id 3118 \
--ID_sampling_method SnP --img_sampling_method 'FPS' --img_sampling_ratio 0.5 \
--output_data '/data/data/VeRi/SnP_3118IDs_0.5Imgs_0610'
```



## Citation

If you find this code useful, please kindly cite:

```
@article{yao2023large,
  title={Large-scale Training Data Search for Object Re-identification},
  author={Yao, Yue and Lei, Huan and Gedeon, Tom and Zheng, Liang},
  journal={arXiv preprint arXiv:2303.16186},
  year={2023}
}
```

If you have any question, feel free to contact yue.yao@anu.edu.au