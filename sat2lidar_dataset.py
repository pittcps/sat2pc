import os
from PIL import Image
import skimage.color
import skimage.io
import skimage.transform
import numpy as np
import torch
import json
import math
from generalized_dataset import GeneralizedDataset
from scipy import ndimage, misc
import random
       
        
class Sat2LidarDataset(GeneralizedDataset):
    def __init__(self, image_dir, ann_dir, mode, pc_num, augment=False):
        super().__init__()

        self.mode = mode
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.image_info = []
        self.ids = []
        self.augment = augment
        self.pc_num = pc_num 
        
        images = [os.path.splitext(name)[0] for name in os.listdir(ann_dir)]

        for i, img_name in enumerate(images):
            self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
        
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": int(image_id),
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        self.ids.append(len(self.ids))

    def reverse_normalizatrion(self, X, ind):
        return (X * self._std[ind]) + self._mean[ind]

    def standard_scale(self, X, mean, std):
        return (X - mean)/std

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def get_image(self, img_id):
        image_id = int(img_id)
        image = skimage.io.imread(self.image_info[image_id]['path'])
        grey_image = self.rgb2gray(image)[..., None]
        
        return grey_image

    def get_annotation(self, image_id):
        info = self.image_info[image_id]
        im_path = info['path']
        file_name =  os.path.splitext(os.path.basename(im_path))[0]
        ann_path = os.path.join(self.ann_dir, file_name + '.json')

        f = open(ann_path)
        ann = json.load(f)
        f.close()
        return ann

    def calc_dis(self, x1, y1, z1, x2, y2, z2):
        return math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2 + (z1-z2) ** 2) 

    def get_distance(self, x, y, z, heights):
        dis = np.zeros(heights.shape)

        for i in range(heights.shape[0]):
            for j in range(heights.shape[1]):
                dis[i][j] = self.calc_dis(x, y, z, i, j, heights[i][j])
        
        return dis

    def load_lidar(self, image_id, rotation = 0, normalized=True):
        ann = self.get_annotation(image_id)
        lidar = ann["lidar"]
        lidar = np.array(lidar, 'float32')
        if lidar.shape[0] == 3:
            lidar = np.transpose(lidar)
        if rotation:
            lidar = self.rotate_lidar(lidar, rotation)
        if normalized == True:
            lidar -= np.mean(lidar, 0, dtype='float64', keepdims=True)
            lidar /= np.std(lidar, 0, dtype='float64', keepdims=True)
        
        return np.array(lidar, 'float32')

    def reverse_lidar_normalization(self, image_id, lidar):
        ann = self.get_annotation(image_id)
        gt_lidar = ann["lidar"]
        gt_lidar = np.array(gt_lidar)
        if gt_lidar.shape[0] == 3:
            gt_lidar = np.transpose(gt_lidar)
        lidar *= np.std(gt_lidar, 0, dtype='float64', keepdims=True)
        lidar += np.mean(gt_lidar, 0, dtype='float64', keepdims=True)
            
        return lidar

    def get_sample_id(self, img_name):
        idx = -1
        for i, info in enumerate(self.image_info):
            if info['id'] == img_name:
                idx = i 

        if idx == -1:
            print('Sample does not exists')
            assert(False)
 
        return idx

    def data_augmentation(self, mat, rotation, is_image = False):
        if is_image == True:
            mat = ndimage.rotate(mat, rotation*90, reshape=False)
        else:
            mat = [ndimage.rotate(m, rotation*90, reshape=False, mode='constant') for m in mat]
        return mat

    def get_init_pc(self):
        Z = np.random.rand(self.pc_num ) + 1.
        h = np.random.uniform(10., 214., size=(self.pc_num ,))
        w = np.random.uniform(10., 214., size=(self.pc_num ,))
        X = (w - 111.5) / 248. * -Z
        Y = (h - 111.5) / 248. * Z
        X = np.reshape(X, (-1, 1))
        Y = np.reshape(Y, (-1, 1))
        Z = np.reshape(Z, (-1, 1))
        XYZ = np.concatenate((X, Y, Z), 1).astype('float32')
        return XYZ

    def __getitem__(self, i):
        if self.augment == True:
            rotation =  random.randint(0, 4)
        else: 
            rotation = 0
        img_id = self.ids[i]
        grey_image = self.get_image(img_id)
        grey_image = self.data_augmentation(grey_image, rotation, True)
        grey_image = (np.transpose(grey_image / 255.0, (2, 0, 1)) - .5) * 2
        target = self.get_target(img_id, rotation)

        init_pc = self.get_init_pc()
        return init_pc, grey_image, target   

    @staticmethod
    def convert_to_xyxy(box): # box format: (xmin, ymin, w, h)
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id, rotation):
        img_id = int(img_id)
        lidar = self.load_lidar(img_id, rotation = rotation)
        lidar = torch.as_tensor(lidar, dtype=torch.float32)
        img_og_id = torch.tensor([self.image_info[img_id]['id']])
        target = dict(image_id=img_og_id, lidar=lidar)
        return target
    
    