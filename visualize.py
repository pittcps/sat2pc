from sat2lidar_dataset import Sat2LidarDataset
import os
import torch
import argparse
import numpy as np
import pptk


import data_util
import utility

def reverse_normalizatrion(X, og_mat):
    mean = np.mean(og_mat)
    std = np.std(og_mat)
    return (X * std) + mean

def run_visualization_with_depth_color(results):
    for res in results:
        for name, result in res.items():

            id = ds.get_sample_id(name)
            _, img, target = ds.__getitem__(id)
        data_util.show_image(img)

        sat2pc_point_cloud = result['final_out']

        gt_point_cloud = ds.load_lidar(id, normalized = False)
        gt_pc_mean = np.mean(gt_point_cloud, 0, keepdims=True)
        gt_pc_std = np.std(gt_point_cloud, 0, keepdims=True)

        gt_point_cloud -= gt_pc_mean
        gt_point_cloud /= gt_pc_std

        gt_point_cloud = utility.remove_padding([torch.tensor(gt_point_cloud).cpu()])[0]

        gt_point_cloud *= gt_pc_std
        gt_point_cloud += gt_pc_mean

        scale = [min(gt_point_cloud[:, 2]), max(gt_point_cloud[:, 2])]

        v1 = pptk.viewer(gt_point_cloud, gt_point_cloud[:, 2])
        v1.color_map('jet', scale = scale)
        v1.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

        sat2pc_point_cloud = utility.remove_padding([torch.tensor(sat2pc_point_cloud).cpu()])[0]
        sat2pc_point_cloud = ds.reverse_lidar_normalization(id, sat2pc_point_cloud)

        v2 = pptk.viewer(sat2pc_point_cloud, sat2pc_point_cloud[:, 2])
        v2.color_map('jet', scale = scale)
        v2.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)
        
        input("Please press enter to visualize the next sample ...")
        v1.close()
        v2.close()


def visualize_results(results, ds, image_dir, show_graphx_res=True, show_final_out=True):
    for res in results:
        for name, result in res.items():
            
            id = ds.get_sample_id(name)
            _, image, target = ds.__getitem__(id)
    
            image = data_util.load_image( os.path.join(image_dir, str(name) + '.png'))

            if show_graphx_res:
                lidar = result['graphx_out']
                rgb = []
                lidar = ds.reverse_lidar_normalization(id, lidar)
                for i in range(len(lidar)):
                    x = int(lidar[i, 0])
                    y = int(lidar[i, 1])
                    rgb.append([0, 1, 0])
                v = pptk.viewer(lidar, np.array(rgb))


            if show_final_out:
                lidar = result['final_out']
                rgb = []
                rgb2 = []

                gt_lidar = ds.load_lidar(id, normalized = False)
                lidar = ds.reverse_lidar_normalization(id, lidar)

                for i in range(len(gt_lidar)):
                    x = int(gt_lidar[i, 0])
                    y = int(gt_lidar[i, 1])
                    rgb2.append([0, 1, 0])
                for i in range(len(lidar)):
                    x = int(lidar[i, 0])
                    y = int(lidar[i, 1])
                    rgb.append([0, 1, 0])

                v = pptk.viewer(gt_lidar, gt_lidar[:, 2])
                v.color_map('jet')
                v.set(point_size=3)
                v = pptk.viewer(lidar, lidar[:, 2])
                v.color_map('jet')
                v.set(point_size=3)

            k = input("Press Enter to continue...")
            if k == 'q':
                print('Stoping the visualization...')
                return

if __name__ == "__main__": 

    parser = argparse.ArgumentParser() 
    parser.add_argument("--data-dir", default="./datasets/extended_dataset/aggregated/splitted")
    parser.add_argument("--results", default="./results/test.res")
    parser.add_argument("--dataset-split", default="test")
    args = parser.parse_args()

    #args.results += args.dataset_split + '.res'

    image_dir = os.path.join(args.data_dir, os.path.join(args.dataset_split, 'image_filtered'))
    ds = Sat2LidarDataset(image_dir = image_dir, pc_num = 1, ann_dir = os.path.join(args.data_dir, os.path.join(args.dataset_split, 'annotation')), mode=args.dataset_split, augment=False)
    
    results = torch.load(args.results, map_location=torch.device('cpu'))

    run_visualization_with_depth_color(results)
    #visualize_results(results, ds, image_dir)
