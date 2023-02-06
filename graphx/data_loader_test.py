import torch as T
from torch.utils.data import Dataset
import numpy as np
import os
import scipy
import json
from PIL import Image


def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h = np.random.uniform(10., 214., size=(num_points,))
    w = np.random.uniform(10., 214., size=(num_points,))
    X = (w - 111.5) / 248. * -Z
    Y = (h - 111.5) / 248. * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')

def init_noisy_pointcloud_loader(num_points, target_pointcloud):
    var = 0.5
    Z = np.random.randn(num_points) * var
    X = np.random.rand(num_points) * var
    Y = np.random.rand(num_points) * var

    idx = np.random.choice(target_pointcloud.shape[0], num_points)

    points = target_pointcloud[idx]

    X = X + points[:, 0]
    Y = Y + points[:, 1]
    Z = Z + points[:, 2]

    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')

def sub_sample_height(h, num_points):

    idx = np.where(h > 0)
    Z = h[idx]
    X = idx[0]
    Y = idx[1]

    idx = np.random.choice(Z.shape[0], num_points)
    X = X[idx]
    Y = Y[idx]
    Z = Z[idx]
    
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def collate(batch):
    data = [b for b in zip(*batch)]
    if len(data) == 4:
        name, init_pc, imgs, gt_pc = data
    elif len(data) == 5:
        name, init_pc, imgs, gt_pc, metadata = data
    else:
        raise ValueError('Unknown data values')

    init_pc = T.from_numpy(np.array(init_pc)).requires_grad_(False)
    imgs = T.from_numpy(np.array(imgs)).requires_grad_(False)
    gt_pc = [T.from_numpy(pc).requires_grad_(False) for pc in gt_pc]
    return (name, init_pc, imgs, gt_pc) if len(data) == 4 else (name, init_pc, imgs, gt_pc, metadata)


class ShapeNet(Dataset):
    def __init__(self, path, grayscale=None, type='train', n_points=2000, **kwargs):
        assert type in ('train', 'val', 'test')
        self.n_points = n_points
        self.grayscale = grayscale
        self.path = os.path.join(path, type)
        self.type = type
        self.files = []

        self.sample_weights = []
        self.image_dir = os.path.join(self.path, "image_filtered")
        self.ann_dir = os.path.join(self.path, "annotation")

        self.files = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(self.ann_dir)]
        self.sample_weights.extend([1/len(self.files)] * len(self.files))

    def rotate_points(self, pc, degree):
        rotation_radians = np.radians(degree)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(pc)
        return rotated_vec

    def load_pytorch_version_results(self, file_name):
        mrcnn_output = os.path.join('./data/mrcnn_output_pytorch/', self.type + '.res')
        results = T.load(mrcnn_output)
        for res in results:
            for name, result in res.items():
                if str(name) == file_name:
                    return result['heights'][0][0].numpy()


    def load_tensroflow_version_results(self, file_name):
        mrcnn_output = os.path.join('./data/mrcnn_output/', file_name + '.json')
        out = open(mrcnn_output,)
        jsn = json.load(out)
        return jsn['heights'][0]

    def load_lidar(self, file_name):
        ann_path = os.path.join(self.ann_dir, file_name + '.json')
        f = open(ann_path)
        ann = json.load(f)
        f.close()
        return np.array(ann["lidar"], 'float32')

    def load_img(self, file_name):
        img_path = os.path.join(self.image_dir, file_name + '.png')
        image = Image.open(img_path)
        image = np.asarray(image)
        image = image[:, :, :3]
        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]
        
        pc = self.load_lidar(file_name)
        if pc.shape[0] == 3:
            pc = pc.T

        img = self.load_img(file_name)
        img = rgb2gray(img)[..., None] if self.grayscale else img
        img = (np.transpose(img / 255.0, (2, 0, 1)) - .5) * 2
        
        pc -= np.mean(pc, 0, dtype='float64', keepdims=True)
        pc /= np.std(pc, 0, dtype='float64', keepdims=True)

        init_points = init_pointcloud_loader(self.n_points)

        return file_name, init_points, np.array(img, 'float32'), pc