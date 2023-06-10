import os
import pathlib
import torchvision
import numpy as np
import torch
from scipy import linalg
from matplotlib.pyplot import imread, imsave
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
import random
import re
import copy
from scipy.special import softmax
from collections import defaultdict
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained
from glob import glob
import os.path as osp
import h5py
import scipy.io
from PIL import Image
import collections
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans
import xml.dom.minidom as XD
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from feat_stas.models.inception import InceptionV3
from feat_stas.strategies import CoreSet
from feat_stas.dataloader import get_id_path_of_data_vehicles
from feat_stas.dataloader import get_id_path_of_data_person
from feat_stas.feat_extraction import get_activations, calculate_frechet_distance, calculate_activation_statistics


def training_set_search(tpaths, data_dict, dataset_id, opt, result_dir, c_num, version):
    """main function of the SnP framework"""

    if version == 'vehicle':
        img_paths,  person_ids,  dataset_ids, meta_dataset  = get_id_path_of_data_vehicles (dataset_id, data_dict)
    if version == 'person':
        img_paths,  person_ids,  dataset_ids, meta_dataset  = get_id_path_of_data_person (dataset_id, data_dict)

    cuda = opt.cuda
    if opt.FD_model == 'inception':
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    if cuda:
        model.cuda()
    batch_size=256

    # caculate the feature stas of target set
    print('=========== caculate the feature stas of target set===========')
    files = []
    if version == 'vehicle':
        if opt.target == 'alice-vehicle':
            for i in range (1, 17):
                target_path = pathlib.Path(tpaths + "/cam" + str(i))
                files.extend(list(target_path.glob('*.jpg')) + list(target_path.glob('*.png')))
        if opt.target == 'veri':
            target_path = pathlib.Path(tpaths)
            files = list(target_path.glob('*.jpg')) + list(target_path.glob('*.png'))
    if version == 'person':
        target_path = pathlib.Path(tpaths)
        files = list(target_path.glob('*.jpg')) + list(target_path.glob('*.png'))


    if not os.path.exists(result_dir + '/target_feature.npy'):
        target_feature = get_activations(opt, files, model, batch_size, dims, cuda, verbose=False)
        np.save(result_dir + '/target_feature.npy', target_feature)
    else:
        target_feature = np.load(result_dir + '/target_feature.npy')
    m1 = np.mean(target_feature, axis=0)
    s1 = np.cov(target_feature, rowvar=False)

    # extracter feature for data pool
    if not os.path.exists(result_dir + '/feature_infer.npy'):
        print('=========== extracting feature of data pool ===========')
        model.eval()
        feature_infer = get_activations(opt, img_paths, model, batch_size, dims, cuda, verbose=False)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        np.save(result_dir + '/feature_infer.npy', feature_infer)
    else:
        feature_infer = np.load(result_dir + '/feature_infer.npy')
        
    person_ids_array=np.array(person_ids)
    mean_feature_per_id=[]
    pid_per_id=[]
    did_per_id=[]

    # get mean fature of perid and the fid, variance of per_id with the target
    if not os.path.exists(result_dir + '/mean_feature_per_id.npy'):
        for did in range(1, len(dataset_id) + 1):
            ind_of_set = np.argwhere(np.array(dataset_ids) == did).squeeze()
            dataset_feature = feature_infer[ind_of_set]
            dataset_pid = person_ids_array[ind_of_set]
            pid_of_dataset=set(dataset_pid)
            for pid in pid_of_dataset:
                ind_of_pid = np.argwhere(np.array(dataset_pid) == pid).squeeze()
                feature_per_id = dataset_feature[ind_of_pid]
                id_ave_feature=feature_per_id.mean(0)
                mean_feature_per_id.append(id_ave_feature)
                pid_per_id.append(pid)
                did_per_id.append(did)
        np.save(result_dir+ '/mean_feature_per_id.npy',mean_feature_per_id)
        pid_did_fid_var = np.c_[np.array(pid_per_id), np.array(did_per_id)]
        np.save(result_dir+ '/pid_did_fid_var.npy', pid_did_fid_var)
    else:
       mean_feature_per_id = np.load(result_dir + '/mean_feature_per_id.npy')
       pid_did_fid_var = np.load(result_dir + '/pid_did_fid_var.npy')

    #remove 0 and -1
    ori_pid_per_id = pid_did_fid_var[:, 0]
    if version == 'vehicle':
        remove_ind=np.r_[np.argwhere(ori_pid_per_id == -1), np.argwhere(ori_pid_per_id == -1)].squeeze()
    else:
        remove_ind=np.r_[np.argwhere(ori_pid_per_id == -1), np.argwhere(ori_pid_per_id == 0)].squeeze()

    new_pid_did_fid_var = np.delete(pid_did_fid_var, remove_ind, 0)
    new_mean_feature_per_id = np.delete(mean_feature_per_id, remove_ind, 0)


    print('\r=========== clustering the data pool ===========')
    pid_per_id = new_pid_did_fid_var[:,0]
    did_per_id = new_pid_did_fid_var[:,1]
    # clustering ids based on ids' mean feature
    if not os.path.exists(result_dir + '/label_cluster_'+str(c_num)+'.npy'):
        # estimator = KMeans(n_clusters=c_num)
        estimator = KMeansConstrained(n_clusters=c_num, size_min=int(np.shape (new_mean_feature_per_id)[0] / c_num ), size_max=int(np.shape (new_mean_feature_per_id)[0] / c_num))
        estimator.fit(new_mean_feature_per_id)
        label_pred = estimator.labels_
        np.save(result_dir + '/label_cluster_'+str(c_num)+'.npy',label_pred)
    else:
        label_pred = np.load(result_dir  + '/label_cluster_'+str(c_num)+'.npy')

    print('\r=========== caculating the fid between T and C_k ===========')
    if not os.path.exists(result_dir + '/cluster_fid_div.npy'):
        cluster_feature = []
        cluster_fid = []
        cluster_div = []
        for k in tqdm(range(c_num)):
            # initializatn of the first seed cluster 0
            initial_pid=pid_per_id[label_pred==k]
            initial_did=did_per_id[label_pred==k]

            initial_feature_infer = feature_infer[(dataset_ids == int(initial_did[0])) & (person_ids_array == initial_pid[0])]
        
            for j in range(1,len(initial_pid)):
                current_feature_infer=feature_infer[(dataset_ids == int(initial_did[j])) & (person_ids_array == initial_pid[j])]
                initial_feature_infer=np.r_[initial_feature_infer, current_feature_infer]
        
            cluster_feature.append(initial_feature_infer)
            
            mu = np.mean(initial_feature_infer, axis=0)
            sigma = np.cov(initial_feature_infer, rowvar=False)

            fea_corrcoef = np.corrcoef(initial_feature_infer)
            fea_corrcoef = np.ones(np.shape(fea_corrcoef)) - fea_corrcoef
            diversity_sum = np.sum(np.sum(fea_corrcoef)) - np.sum(np.diagonal(fea_corrcoef))
            current_div = diversity_sum / (np.shape (fea_corrcoef)[0] ** 2 - np.shape (fea_corrcoef)[0])

            # caculating domain gap
            current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
            cluster_fid.append(current_fid)
            cluster_div.append(current_div)
            # cluster_mmd.append(current_mmd)
        np.save(result_dir + '/cluster_fid_div.npy', np.c_[np.array(cluster_fid), np.array(cluster_div)])
        #np.save(result_dir+'/cluster_fid_var.npy', np.c_[np.array(cluster_fid),np.array(cluster_var_gap)])
    else:
        cluster_fid_var=np.load(result_dir + '/cluster_fid_div.npy')
        cluster_fid=cluster_fid_var[:,0]
        cluster_div=cluster_fid_var[:,1]

    cluster_fida=np.array(cluster_fid)
    score_fid = softmax(-cluster_fida)
    sample_rate = score_fid

    c_num_len = []
    for kk in range(c_num):
        initial_pid = pid_per_id[label_pred == kk]
        c_num_len.append(len(initial_pid))

    id_score = []
    for jj in range(len(label_pred)):
        id_score.append(sample_rate[label_pred[jj]] / c_num_len[label_pred[jj]])

    if opt.ID_sampling_method == 'random':
        selected_data_ind = np.sort(np.random.choice(range(len(id_score)), opt.n_num_id, replace=False))
    if opt.ID_sampling_method == 'SnP':
        lowest_fd = float('inf')
        lowest_id_list = []
        if not os.path.exists(result_dir + '/domain_seletive_ids.npy'):
            cluster_rank = np.argsort(cluster_fida)
            current_list = []
            cluster_feature_aggressive = []
            for k in tqdm(cluster_rank):
                id_list = np.where (label_pred==k)[0]
                initial_pid=pid_per_id[label_pred==k]
                initial_did=did_per_id[label_pred==k]
                initial_feature_infer = feature_infer[(dataset_ids == int(initial_did[0])) & (person_ids_array == initial_pid[0])]
                for j in range(1,len(initial_pid)):
                    current_feature_infer=feature_infer[(dataset_ids == int(initial_did[j])) & (person_ids_array == initial_pid[j])]
                    initial_feature_infer=np.r_[initial_feature_infer, current_feature_infer]

                cluster_feature_aggressive.extend(initial_feature_infer)
                cluster_feature_aggressive_fixed = cluster_feature_aggressive
                target_feature_fixed = target_feature
                if len (cluster_feature_aggressive) > len (target_feature):
                    cluster_idx = np.random.choice(range(len (cluster_feature_aggressive)), len(target_feature), replace=False)
                    cluster_feature_aggressive_fixed = np.array([cluster_feature_aggressive[ii] for ii in cluster_idx])
                if len (cluster_feature_aggressive) < len (target_feature):
                    cluster_idx = np.random.choice(range(len(target_feature)), len (cluster_feature_aggressive), replace=False)
                    target_feature_fixed = target_feature[cluster_idx]
                mu = np.mean(cluster_feature_aggressive_fixed, axis=0)
                sigma = np.cov(cluster_feature_aggressive_fixed, rowvar=False)
                current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
                current_list.extend (list (id_list))
                print (lowest_fd, current_fid)
                if lowest_fd > current_fid:
                    lowest_fd = current_fid
                    lowest_id_list = copy.deepcopy(current_list)
            np.save(result_dir + '/domain_seletive_ids.npy', lowest_id_list)
        else:
            lowest_id_list = np.load(result_dir + '/domain_seletive_ids.npy')
        print ("searched IDs", len (lowest_id_list))
        direct_selected_data_ind = np.array(lowest_id_list)
        if opt.n_num_id < len(direct_selected_data_ind):
            selected_data_ind = np.sort(np.random.choice(direct_selected_data_ind, opt.n_num_id, replace=False))
        else:
            selected_data_ind = np.array(lowest_id_list)
        
    if opt.ID_sampling_method == 'greedy':
        selected_data_ind = np.argsort(id_score)[-opt.n_num_id:]


    sdid = did_per_id[selected_data_ind]
    spid = pid_per_id[selected_data_ind]
    print (collections.Counter(sdid))
    data_dir = result_dir + '/proxy_set'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    print('\r=========== building training set ===========')
    sampled_data=np.c_[sdid,spid]
    if not os.path.exists(result_dir + '/'  + str(opt.ID_sampling_method) + '_' + str(opt.n_num_id) + '_result_IMG_list_'+str(c_num)+'.npy'):
        result_IMG_list, one_img_perID = IDidx2IMGidx(data_dict, dataset_id, sampled_data, opt.output_data, meta_dataset, feature_infer, opt)
        np.save(result_dir + '/'  + str(opt.ID_sampling_method) + '_' + str(opt.n_num_id) + '_result_IMG_list.npy', result_IMG_list)
        np.save(result_dir + '/'  + str(opt.ID_sampling_method) + '_' + str(opt.n_num_id) + '_one_img_perID.npy', one_img_perID)
    else:
        result_IMG_list = np.load(result_dir + '/'  + str(opt.ID_sampling_method) + '_' + str(opt.n_num_id) + '_result_IMG_list.npy')
        one_img_perID = np.load(result_dir + '/'  + str(opt.ID_sampling_method) + '_' + str(opt.n_num_id) + '_one_img_perID.npy')
    
    result_IMG_list = np.array(result_IMG_list)
    one_img_perID = np.array(one_img_perID)

    if opt.img_sampling_ratio < 1:
        if opt.img_sampling_method == 'FPS':
            direct_selected_data_ind = np.array(result_IMG_list)
            idb_is = np.zeros(len(direct_selected_data_ind), dtype=bool)
            for ii in one_img_perID:
                index_lowest_first_img_per_id = list(result_IMG_list).index(ii)
                idb_is[index_lowest_first_img_per_id] = 1
            selected_data_feature = feature_infer [direct_selected_data_ind]
            strategy = CoreSet (None, np.zeros(len(direct_selected_data_ind), dtype=bool), np.zeros(len(direct_selected_data_ind), dtype=bool), None, None, None)
            img_num = int (opt.img_sampling_ratio * len (result_IMG_list))
            div_selected_data_ind = strategy.query (img_num - len(one_img_perID), selected_data_feature, idb_is)
            selected_img_ind = list(one_img_perID)
            selected_img_ind.extend (direct_selected_data_ind[div_selected_data_ind])
        
        if opt.img_sampling_method == 'random':
            lowest_img_list = list(result_IMG_list)
            for ii in one_img_perID:
                lowest_img_list.remove (ii)
            direct_selected_data_ind = np.array(lowest_img_list)
            img_num = int (opt.img_sampling_ratio * len (result_IMG_list))
            if img_num > len(one_img_perID):
                selected_img_ind = list(np.sort(np.random.choice(direct_selected_data_ind, img_num - len(one_img_perID), replace=False)))
            else:
                selected_img_ind = []
            selected_img_ind.extend (one_img_perID)
    else:
        selected_img_ind = result_IMG_list

    result_feature = dataset_build_img(data_dict, dataset_id, selected_img_ind, data_dir, meta_dataset, feature_infer, opt)

    mu = np.mean(result_feature, axis=0)
    sigma = np.cov(result_feature, rowvar=False)
    current_fid = calculate_frechet_distance(m1, s1, mu, sigma)
   
    print('finished with a dataset has FD', current_fid, "to the target")
    return sampled_data


def IDidx2IMGidx(dict, dataset_id, sampled_data, result_dir, meta_dataset, feature_infer, opt):
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    pid_sampled = sampled_data[:, 1]  
    did_sampled = sampled_data[:, 0]

    all_pids = {}
    image_searched = []
    one_img_perID = []
    id_count = 0

    for idx, (fname, pid, did, cid) in enumerate(meta_dataset):
        exist_judge = False
        for j in range (len(pid_sampled)):
            if str(pid) == str(pid_sampled[j]) and str(did) == str(did_sampled[j]): 
                exist_judge = True
        if exist_judge:
            if pid not in all_pids:
                all_pids[pid] = {}
                all_pids[pid][did] = id_count
                one_img_perID.append (idx)
                id_count += 1
            if pid in all_pids:
                if did not in all_pids[pid]:
                    all_pids[pid][did] = id_count
                    one_img_perID.append (idx)
                    id_count += 1

            image_searched.append (idx)
    print ("id count", id_count, "img count", len (image_searched))
    return image_searched, one_img_perID

def dataset_build_img(dict, dataset_id, sampled_data, result_dir, meta_dataset, feature_infer, opt):
    all_pids = {}
    dstr_path = opt.output_data
    if not os.path.isdir(dstr_path):
        os.mkdir(dstr_path)
    img_count = 0
    id_count = 0
    feature_searched = []
    for idx in sampled_data:
        fname, pid, did, cid = meta_dataset[idx]
        if pid not in all_pids:
            all_pids[pid] = {}
            all_pids[pid][did] = id_count
            id_count += 1
        if pid in all_pids:
            if did not in all_pids[pid]:
                all_pids[pid][did] = id_count
                id_count += 1
        feature_searched.append (idx)
        pid = all_pids[pid][did]
        img_count += 1
        # print (pid, cid, did)
        new_path = dstr_path + '/' + '{:04}'.format(pid) + "_c" + '{:03}'.format(cid) + "_d" + '{:03}'.format(did) + "_" + str(img_count) + '.jpg'
        if opt.no_sample:
            continue
        copyfile(fname, new_path)

    print ("successfully create a dataset with", img_count, "images, and", id_count, "ids")
    feature_new = feature_infer[feature_searched]
    return feature_new
