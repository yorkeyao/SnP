import os
from feat_stas import SnP
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--result_dir', type=str, metavar='PATH', default='sample_data/')
parser.add_argument('--c_num', default=50, type=int, help='number of cluster')
parser.add_argument('--n_num_id', default=301, type=int, help='number of ids')
parser.add_argument('--target', type=str, default='market', choices=['market', 'alice-person'], help='items used to caculate sampling score')
parser.add_argument('--output_data', type=str, metavar='PATH', default='/data/reid_data/alice-person/random_301')
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
        'duke': '/data/reid_data/duke_reid/bounding_box_train/', #duke
        'market': '/data/reid_data/market/bounding_box_train/', #market
        'msmt': '/data/reid_data/MSMT/MSMT_bounding_box_train/', #msmt
        'cuhk': '/data/reid_data/cuhk03_release/', #cuhk
        'alice-person': '/data/reid_data/alice-person/bounding_box_train/', #alice
        'raid': '/data/reid_data/RAiD_Dataset-master/', #raid
        'unreal': '/data/reid_data/unreal/UnrealPerson-data/unreal_v3.1/images/', #unreal
        'personx': '/data/reid_data/personx/bounding_box_train/',  #personx
        'randperson': '/data/reid_data/randperson_subset/randperson_subset/', #randperson
        'pku': '/data/reid_data/PKU-Reid/PKUv1a_128x48/', #pku
        'ilids': '/data/reid_data/i-LIDS-VID/', #ilids
        'viper': '/data/reid_data/VIPeR/', # viper 
        }


databse_id = ['duke', 'market', 'msmt', 'cuhk', 'raid', 'unreal', 'personx', 'randperson', 'pku', 'viper']

# databse_id = ['duke', 'market', 'msmt'] # pool_A
# databse_id = ['duke', 'market', 'msmt', 'unreal'] # pool_B
# databse_id = ['duke', 'market', 'msmt', 'unreal', 'personx', 'randperson'] # pool_C

if opt.target == 'alice-person':
    target = data_dict['alice-person'] 
if opt.target == 'market':
    target = data_dict['market'] 
    databse_id.remove ('market')

result_dir=opt.result_dir

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

if os.path.isdir(opt.output_data):
    assert ("output dir has already exist")

SnP.training_set_search(target, data_dict, databse_id, opt, result_dir, c_num, version = "person")


