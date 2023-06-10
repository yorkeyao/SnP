import os
import torchvision
import numpy as np
from matplotlib.pyplot import imread, imsave
import random
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glob import glob
import os.path as osp
import h5py
import scipy.io
from PIL import Image
import numpy as np
import xml.dom.minidom as XD




def makeDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def process_veri_wild_vehicle(root, vehicle_info):
    imgid2vid = {}
    imgid2camid = {}
    imgid2imgpath = {}
    vehicle_info_lines = open(vehicle_info, 'r').readlines()

    for idx, line in enumerate(vehicle_info_lines):
        # if idx < 10:
        vid = line.strip().split('/')[0]
        imgid = line.strip().split(';')[0].split('/')[1]
        camid = line.strip().split(';')[1]
        img_path = osp.join(root, 'images', imgid + '.jpg')
        imgid2vid[imgid] = vid
        imgid2camid[imgid] = camid
        imgid2imgpath[imgid] = img_path

    assert len(imgid2vid) == len(vehicle_info_lines)
    return imgid2vid, imgid2camid, imgid2imgpath

def get_id_path_of_data_vehicles (dataset_id, data_dict):
    """create data loader for vehicle re-ID datsets"""
    img_paths = []
    dataset_ids = []
    person_ids = []
    ret = []
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    total_ids = 0

    # for each dataset in candidate list:
    for dd_idx, dataset in enumerate(dataset_id):
        all_pids = {}
        last_img_num = len (img_paths)
        if dataset == 'veri':
            root = data_dict['veri'] + '/image_train/'
            fpaths = sorted(glob(osp.join(root, '*.jpg'))) 
            dataset_id_list = [dd_idx + 1 for n in range(len(fpaths))]
            dataset_ids.extend (dataset_id_list)
            for fpath in fpaths:
                fname = root + osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                
                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, dd_idx + 1, cam))
        
        if dataset == 'aic': 
            root = data_dict['aic']
            train_path = osp.join(root, 'image_train')
            xml_dir = osp.join(root, 'train_label.xml')
            reid_info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
            index_by_fname_dict = defaultdict()
            for index in range(len(reid_info)):
                fname = reid_info[index].getAttribute('imageName')
                index_by_fname_dict[fname] = index

            fpaths = sorted(glob(osp.join(train_path, '*.jpg')))
            dataset_id_list = [dd_idx + 1 for n in range(len(fpaths))]
            dataset_ids.extend (dataset_id_list)
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, [reid_info[index_by_fname_dict[fname]].getAttribute('vehicleID'),
                                    reid_info[index_by_fname_dict[fname]].getAttribute('cameraID')[1:]])
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                cam -= 1
                fname = train_path + "/" + fname
                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, dd_idx + 1, cam))

        if dataset == 'vid':
            root = data_dict['vid']
            label_path = osp.join (root, 'train_test_split')
            train_path_label = osp.join(label_path, 'train_list.txt')
            
            with open(train_path_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                for line in lines:
                    fname, pid = line
                    fname = fname + ".jpg"
                    if pid == -1: continue
                    fname = root + "image/" + fname
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids)
                    pid = all_pids[pid]
                    img_paths.append(fname)
                    person_ids.append(pid)
                    ret.append((fname, pid, dd_idx + 1, cam))
                    dataset_ids.append (dd_idx + 1)
        
        if dataset == 'vehiclex':
            root = data_dict['vehiclex']
            fpaths = sorted(glob(osp.join(root, '*.jpg')))
            dataset_id_list = [dd_idx + 1 for n in range(len(fpaths))]
            dataset_ids.extend (dataset_id_list)
            random.shuffle(fpaths)
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                fname = root + fname
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, dd_idx + 1, cam))
            
        if dataset == 'veri-wild':
            root = data_dict['veri-wild']
            train_list = osp.join(root, 'train_test_split/train_list_start0.txt')
            vehicle_info = osp.join(root, 'train_test_split/vehicle_info.txt')
            imgid2vid, imgid2camid, imgid2imgpath = process_veri_wild_vehicle(root, vehicle_info)
            vid_container = set()
            img_list_lines = open(train_list, 'r').readlines()
            for idx, line in enumerate(img_list_lines):
                line = line.strip()
                vid = line.split('/')[0]
                vid_container.add(vid)
            vid2label = {vid: label for label, vid in enumerate(vid_container)}

            dataset_id_list = [dd_idx + 1 for n in range(len(img_list_lines))]
            dataset_ids.extend (dataset_id_list)

            for idx, line in enumerate(img_list_lines):
                line = line.strip()
                pid = int(line.split('/')[0])
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]

                imgid = line.split('/')[1].split('.')[0]
                # if relabel: vid = vid2label[vid]
                img_paths.append(imgid2imgpath[imgid])
                person_ids.append(pid)
                # print ((imgid2imgpath[imgid], int(vid), 5, int(imgid2camid[imgid])))
                ret.append((imgid2imgpath[imgid], pid, dd_idx + 1, int(imgid2camid[imgid])))
        
        if dataset == 'stanford_cars':
            root = data_dict['stanford_cars']
            stanford_dataset = torchvision.datasets.StanfordCars(root=root, download=True)

            dataset_id_list = [dd_idx + 1 for n in range(len(stanford_dataset))]
            dataset_ids.extend (dataset_id_list)

            for fname, pid in stanford_dataset._samples:
                img_paths.append(fname)
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                person_ids.append(int(pid))
                ret.append((fname, int(pid), dd_idx + 1, 1))
        
        if dataset == 'vd1':
            root = data_dict['vd1']
            label_path = osp.join (root, 'train_test')
            train_path_label = osp.join(label_path, 'trainlist.txt')
            
            with open(train_path_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                for line in lines:
                    fname, pid, _, _ = line
                    fname = fname + ".jpg"
                    if pid == -1: continue
                    fname = root + "image/" + fname
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids)
                    pid = all_pids[pid]
                    img_paths.append(fname)
                    person_ids.append(pid)
                    ret.append((fname, pid, dd_idx + 1, cam))
                    dataset_ids.append (dd_idx + 1)
        
        if dataset == 'vd2':
            root = data_dict['vd2']
            label_path = osp.join (root, 'train_test')
            train_path_label = osp.join(label_path, 'trainlist.txt')
            
            with open(train_path_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                for line in lines:
                    fname, pid, _, _ = line
                    fname = fname + ".jpg"
                    if pid == -1: continue
                    fname = root + "image/" + fname
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids)
                    pid = all_pids[pid]
                    img_paths.append(fname)
                    person_ids.append(pid)
                    ret.append((fname, pid, dd_idx + 1, cam))
                    dataset_ids.append (dd_idx + 1)
        

        total_ids += len (all_pids)
        print("ID", dd_idx + 1, "dataset", dataset, "loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
                .format(len(all_pids), len(img_paths) - last_img_num))

    print ("whole dataset contains", total_ids, "ids", len (img_paths), "images")
    return img_paths, person_ids, np.array(dataset_ids), ret

def get_id_path_of_data_person (dataset_id, data_dict):
    """create data loader for person re-ID datsets"""
    img_paths = []
    dataset_ids = []
    person_ids = []
    ret = []
    total_ids = 0

    for idx, dataset in enumerate(dataset_id):
        all_pids = {}
        last_img_num = len (img_paths)
        if dataset in ['duke', 'market', 'msmt', 'unreal', 'personx', 'randperson', 'pku']:
            root = data_dict[dataset]
            fpaths = sorted(glob(osp.join(root, '*.jpg')) + glob(osp.join(root, '*.png'))) 

            dataset_id_list = [idx + 1 for n in range(len(fpaths))]
            dataset_ids.extend (dataset_id_list)

            pattern = re.compile(r'([-\d]+)_c([-\d]+)')
            if dataset == 'randperson':
                pattern = re.compile(r'([-\d]+)_s([-\d]+)_c([-\d]+)')
            if dataset == 'pku':
                pattern = re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')

            
            for fpath in fpaths:
                fname = root + osp.basename(fpath)
                if fname.endswith('.png'):
                    Image.open(fname).save(fname.split('.')[0] + '.jpg')
                    fname = fname.split('.')[0] + '.jpg'
                pid, cam = 0, 0
                if dataset == 'randperson':
                    pid, sid, cam = map(int, pattern.search(fname).groups())
                elif dataset == 'pku':
                    pid, sid, cnt_num = map(int, pattern.search(fname).groups())
                else:
                    pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, idx + 1, cam))

        if dataset == 'raid':
            root = data_dict[dataset]
            raid = h5py.File(os.path.join(root, 'RAiD_4Cams.mat'))
            images = raid['dataset']['images']
            camID = raid['dataset']['cam']
            labels = raid['dataset']['personID']

            dataset_id_list = [idx + 1 for n in range(len(images))]
            dataset_ids.extend (dataset_id_list)
            images_dst_path = os.path.join(root, "train_all")

            for idx_img in range (len(images)):
                np_image = images[idx_img].T
                img = Image.fromarray(np_image)
                pid = labels[idx_img][0]
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                cid = camID[idx_img][0]
                if not os.path.isdir(images_dst_path):
                    os.mkdir(images_dst_path)                
                fname = images_dst_path + '/' + id_label + '_' + 'c' + \
                    str(cid) + '_' + str(idx_img).zfill(5) + '.jpg'

                if not os.path.exists(fname):
                    img.save(os.path.join(img_dst_path, fname))

                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, idx + 1, cid))
            
        if dataset == 'cuhk':
            '''
             download "cuhk03_new_protocol_config_detected.mat" from "https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03"
            '''
            root = data_dict[dataset]
            cuhk03 = h5py.File(os.path.join(root, 'cuhk-03.mat'))
            config = scipy.io.loadmat(os.path.join(
                root, 'cuhk03_new_protocol_config_detected.mat'))
            train_idx = config['train_idx'].flatten()
            # gallery_idx = config['gallery_idx'].flatten()
            # query_idx = config['query_idx'].flatten()
            labels = config['labels'].flatten()
            filelist = config['filelist'].flatten()
            cam_id = config['camId'].flatten()

            imgs = cuhk03['detected'][0]
            cam_imgs = []
            for i in range(len(imgs)):
                cam_imgs.append(cuhk03[imgs[i]][:].T)

            images_dst_path = os.path.join(root, "train_all")
            makeDir(images_dst_path)

            dataset_id_list = [idx + 1 for n in range(len(train_idx))]
            dataset_ids.extend (dataset_id_list)

            for i in train_idx:
                i -= 1  # Start from 0
                file_name = filelist[i][0]
                cam_pair_id = int(file_name[0])
                cam_label = int(file_name[2: 5])
                cam_image_idx = int(file_name[8: 10])

                np_image = cuhk03[cam_imgs[cam_pair_id - 1]
                                [cam_label - 1][cam_image_idx - 1]][:].T

                unified_cam_id = (cam_pair_id - 1) * 2 + cam_id[i]
                img = Image.fromarray(np_image)

                pid = labels[i]
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]

                id_label = str(labels[i]).zfill(4)
                img_dst_path = os.path.join(images_dst_path, id_label)

                # If the dir not exists yet, save this first image to val set
                if not os.path.isdir(img_dst_path):
                    os.mkdir(img_dst_path)

                fname = root + id_label + '_' + 'c' + \
                    str(unified_cam_id) + '_' + str(cam_image_idx).zfill(2) + '.jpg'
                if not os.path.exists(fname):
                    img.save(os.path.join(img_dst_path, fname))

                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, idx + 1, unified_cam_id))
        
        if dataset == 'viper':
            root = data_dict[dataset]
            images_dir = osp.join(root, 'images/')
            makeDir(images_dir)
            cameras = [sorted(glob(osp.join(root, 'cam_a', '*.bmp'))),
                    sorted(glob(osp.join(root, 'cam_b', '*.bmp')))]
            assert len(cameras[0]) == len(cameras[1])

            for pid, (cam1, cam2) in enumerate(zip(*cameras)):
                
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                # view-0
                fname = images_dir + '{:08d}_c{:02d}_{:04d}.jpg'.format(pid, 0, 0)
                imsave(osp.join(images_dir, fname), imread(cam1))

                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, idx + 1, 1))

                # view-1
                fname = images_dir + '{:08d}_c{:02d}_{:04d}.jpg'.format(pid, 1, 0)
                imsave(osp.join(images_dir, fname), imread(cam2))

                img_paths.append(fname)
                person_ids.append(pid)
                ret.append((fname, pid, idx + 1, 1))

            dataset_id_list = [idx + 1 for n in range(len(img_paths) - last_img_num)]
            dataset_ids.extend (dataset_id_list)
            

        total_ids += len (all_pids)
        print("dataset", dataset, "loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
                .format(len(all_pids), len(img_paths) - last_img_num))

    print ("whole dataset contains", total_ids, "ids", len (img_paths), "images")
    return img_paths, person_ids, np.array(dataset_ids), ret