import os
from feat_stas import SnP
import argparse
import numpy as np
import logging

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--result_dir', type=str, metavar='PATH', default='sample_data/')
parser.add_argument('--c_num', default=50, type=int, help='number of cluster')
parser.add_argument('--n_num_id', default=2000, type=int, help='number of ids')
parser.add_argument('--target', type=str, default='veri', choices=['veri', 'alice-vehicle'], help='items used to caculate sampling score')
parser.add_argument('--output_data', type=str, metavar='PATH', default='/data/reid_data/alice-vehicle/searched_2000(3000)_directly_6data')
parser.add_argument('--ID_sampling_method', type=str, default='SnP', choices=['greedy', 'random', 'SnP'], help='how to sample')
parser.add_argument('--img_sampling_ratio', default=1.0, type=float, help='image sampling ratio')
parser.add_argument('--img_sampling_method', type=str, default='FPS', choices=['FPS', 'random'], help='how to sample')
parser.add_argument('--no_sample', action='store_true', help='do not perform sample')
parser.add_argument('--cuda', action='store_true', help='whether cuda is enabled')
parser.add_argument('--FD_model', type=str, default='inception', choices=['inception', 'posenet'],
                    help='model to calculate FD distance')

opt = parser.parse_args()
result_dir=opt.result_dir
c_num=opt.c_num
n_num_id=opt.n_num_id

np.random.seed(0)

# data pool
data_dict = {
        'veri': '/data/reid_data/VeRi/',
        'aic': '/data/reid_data/AIC19-reid/', 
        'alice-vehicle': '/data/reid_data/alice-vehicle/', 
        'vid': '/data/reid_data/VehicleID_V1.0/',
        'vehiclex': '/data/reid_data/alice-vehicle/vehicleX_random_attributes/',
        'veri-wild': '/data/reid_data/veri-wild/VeRI-Wild/',
        'stanford_cars': '/data/reid_data/stanfordcar/',
        'compcars': '/data/reid_data/compcar/CompCars/',
        'vd1': '/data/reid_data/PKU-VD/PKU-VD/VD1/',
        'vd2': '/data/reid_data/PKU-VD/PKU-VD/VD2/'
        }

databse_id= ['veri', 'aic', 'vid', 'veri-wild', 'vehiclex', 'stanford_cars', 'vd1', 'vd2']

if opt.target == 'alice-vehicle':
    target = data_dict['alice-vehicle'] + 'alice-vehicle_train'
if opt.target == 'veri':
    target = data_dict['veri'] + "image_train"
    databse_id.remove ('veri')


result_dir=opt.result_dir

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

if os.path.isdir(opt.output_data):
    assert ("output dir has already exist")

SnP.training_set_search(target, data_dict, databse_id, opt, result_dir, c_num, version = "vehicle")


