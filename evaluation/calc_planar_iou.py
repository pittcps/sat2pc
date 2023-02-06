from code import interact
from random import sample
import sys
import os
import numpy as np
import random
import pptk
import matplotlib.pyplot as plt
import math
from itertools import repeat
from sympy import Plane, Point3D, N
import pyransac3d as pyrsc
import json
from .RoofPlaneSementation import prepare_result_data
from .RoofPlaneSementation import prepare_graphx_results
from .RoofPlaneSementation import run_on_dataset
sys.path.append("../")
import data_util


def get_z(x, y, eq):
    z = -(eq[0]*x + eq[1]*y + eq[3])/eq[2]
    return z

def fit_plane_ransac(plane_points, dense_std, dense_mean, debug):
    plane_eq = []
    sympy_planes = {}
    if debug:
        _lidar = []
        rgb = []

    # Find points for each segment
    for key, points in plane_points.items():
        points = np.asarray(points)
        points = (points[:, :3] * dense_std) + dense_mean
        if len(points) < 3:
            plane_eq.append([])
            print('found empty plane!!')
            continue

        plane = pyrsc.Plane()
        eq, point_idx = plane.fit(points, thresh = 0.5, maxIteration=5000)#.equation.reshape(-1)  # thresh = 0.5
        plane_eq.append(eq)

        if eq[2] == 0:
            continue

        while True:

            sample_points = []

            for i in range(3):
                rand_ind = random.randint(0, points.shape[0] - 1)
                z = get_z(points[rand_ind][0], points[rand_ind][1], eq)
                sample_points.append(Point3D(points[rand_ind][0], points[rand_ind][1], z))
            
            try:
                sympy_plane = Plane(sample_points[0], sample_points[1], sample_points[2])
                sympy_planes[key] = sympy_plane

            except ValueError:
                continue
            break

        if debug:
            fitted_points = []
            for x, y, _ in points:
                z = get_z(x, y, eq)
                fitted_points.append([x, y, z])
            _lidar = _lidar + fitted_points
            print(len(_lidar))
            rgb.extend(repeat([float(i)/255 for i in key.split()], len(points)))

    if debug:
        _lidar = np.asarray(_lidar)
        print(_lidar.shape)
        #_lidar[:, :3] = (_lidar[:, :3] * dense_std) + dense_mean
        _lidar = np.asarray(_lidar).T
        data_util.plot_3d(_lidar, len(_lidar[0]), rgb)
    return sympy_planes

def extract_planes(data):
    planes = {}
    for i in range(data.shape[0]):
        cl = "{} {} {}".format(data[i, 3], data[i, 4], data[i, 5])
        if cl not in planes:
            planes[cl] = []
        planes[cl].append(data[i, :3])

    # filtering out small planes 
    removed_plane = 0
    final_planes = {}
    for key, p in planes.items():
        if len(p) > 15:
            final_planes[key] = p
        else: 
            removed_plane = removed_plane + 1
    #print('removed planes: ', removed_plane)
    return final_planes

def get_boundaries(planes, l = 5.5):
    boundaries = {}
    l = 1
    for key, p in planes.items():
        p = np.asarray(p)
        bndry = data_util.get_pointcolud_outline(p, l = l)
        boundaries[key] = [bndry, len(p)]

    return boundaries

def plot_two_boundaries(pp, gp):
    fig, axs = plt.subplots()
    axs.fill(pp[:, 0], pp[:, 1], alpha=0.5, fc='r', ec='none')
    axs.fill(gp[:, 0], gp[:, 1], alpha=0.5, fc='b', ec='none')
    plt.show()

'''
boundaries1: ground trush
boundaries2: predictions

'''

def get_match(boundaries1, boundaries2, threshold = 0.01):
    final_matches = []
    tp = fp = fn = 0
    updated = True
    while updated:
        updated = False
        matches = []
        for key1, v in boundaries1.items():
            best_match = ''
            best_iou = 0    
            bounday1 = v[0]
            points1 = v[1]    
            for key2, v2 in boundaries2.items():
                bounday2 = v2[0]
                points2 = v2[1]

                intersect = bounday1.intersection(bounday2).area
                union = bounday1.union(bounday2).area
                iou = intersect / union

                intersect_percentage = intersect/bounday1.area
                if intersect_percentage > threshold and iou > best_iou:
                    best_iou = iou
                    best_match = key2
            
            matches.append([key1, best_match, best_iou, points1])

        matches.sort(key=lambda x:x[2], reverse=True)

        for x in matches:
            if x[1] in boundaries2:
                updated = True
                final_matches.append(x)
                boundaries2.pop(x[1], None)
                boundaries1.pop(x[0], None)
                tp += 1
                
    for key1, v in boundaries1.items():
        points1 = v[1]  
        final_matches.append([key1, '', 0, points1])
        fn += 1
    fp = len(boundaries2)
    return final_matches, tp, fp, fn
       
def create_lidar_rgb(plane_dict, matches):
    lidar = []
    rgb = []
    for key, v in plane_dict.items():
        lidar.extend(v)
        if matches == None:
            color = [float(i)/255 for i in key.split()]
        else:
            color = [1, 1, 1]
            for x in matches:
                if x[1] == key:
                    color = [float(i)/255 for i in x[0].split()]
                    break
        rgb.extend(repeat(color,len(v)))

    return lidar, rgb

def visualize_3d(planes, matches = None):
    lidar, rgb = create_lidar_rgb(planes, matches)
    
    #lidar = data[:, :3] 
    #rgb = data[:, 3:]/255.
    v = pptk.viewer(lidar, rgb)
    v.set(point_size=10)

def calc_tilt(sympy_planes):
    tilts = {}
    xy_plane = Plane(Point3D(10, 10, 0), Point3D(10, 0, 0), Point3D(0, 10, 0))
    for key, plane in sympy_planes.items():
        out = plane.angle_between(xy_plane)
        tilt = N(out) * (180/math.pi)
        if tilt > 90:
            tilt = 180 - tilt
        tilts[key] = tilt
    return tilts

def calc_tilt_error(pred_tilts, gt_tilts, pred_to_gt_match):
    error = {}

    for key, v in pred_tilts.items():
        if key in pred_to_gt_match.keys():
            matched_gt_key = pred_to_gt_match[key]
            error[key] = gt_tilts[matched_gt_key] - pred_tilts[key]

    return error

def calc_area_difference_percentage_over_gt_area(gt_plane_boundaries, pred_plane_boundaries, pred_to_gt_match):
    diffs = {}
    for key, val in pred_plane_boundaries.items():
        if key in pred_to_gt_match.keys():
            pred_area = val[0].area
            gt_area = gt_plane_boundaries[pred_to_gt_match[key]][0].area
            diffs[key] = ((gt_area - pred_area)/gt_area)*100
            #print("Gt area={}, pred area={}, diff={}".format(gt_area, pred_area, gt_area - pred_area))
    return diffs


def calc_planar_metrics(dataset_dir, results_dir, prepared_prediction_data_dir, segmentation_results_dir, segmented_gt_planes_dir, metric_saving_dir, model, debug):

    image_dir = os.path.join(dataset_dir, "image_filtered")
    ann_dir = os.path.join(dataset_dir, "annotation") 
    dense_ann_dir = os.path.join(dataset_dir, "annotation_dense") 
    exe_path = '..\RoofPlaneSementation'
    os.makedirs(prepared_prediction_data_dir, exist_ok=True)
    os.makedirs(segmentation_results_dir, exist_ok=True)

    if model == "graphx":
        prepare_graphx_results.prepare_data(image_dir, ann_dir, results_dir, prepared_prediction_data_dir)
        run_on_dataset.run_segmentation_algo(exe_path, prepared_prediction_data_dir, segmentation_results_dir, th = 0.005)
    elif model == "sat2pc" or model == "psgn":
        # prepare the predictions for segmentation algorithm
        prepare_result_data.prepare_data(image_dir, ann_dir, results_dir, prepared_prediction_data_dir)
        run_on_dataset.run_segmentation_algo(exe_path, prepared_prediction_data_dir, segmentation_results_dir, th = 0.013)

    samples = [os.path.splitext(name)[0] for name in os.listdir(segmented_gt_planes_dir)]
    average_plane_absolute_tilt_errors = []
    average_absolute_plane_area_diff = []
    all_complt = []
    all_corr = []
    all_qual = []

    for name in samples:
        gt_data = np.loadtxt(os.path.join(segmented_gt_planes_dir, name + ".txt"))
        pred_data = np.loadtxt(os.path.join(segmentation_results_dir, name + ".txt"))
        gt_data[:, :3] = gt_data[:, :3] * 100
        pred_data[:, :3] = pred_data[:, :3] * 100

        og_lidar = np.asarray(data_util.load_json(os.path.join(ann_dir, name + '.json'))['lidar']).T

        og_mean = np.mean(og_lidar, 0, dtype='float64', keepdims=True)
        og_std = np.std(og_lidar, 0, dtype='float64', keepdims=True)

        og_lidar -= og_mean
        og_lidar /= og_std

        dense =  np.asarray(data_util.load_json(os.path.join(dense_ann_dir, name + '.json'))['lidar']).T
        dense_mean = np.mean(dense, 0, dtype='float64', keepdims=True)
        dense_std = np.std(dense, 0, dtype='float64', keepdims=True)

        gt_data[:, :3] = (gt_data[:, :3] - dense_mean)/dense_std
        pred_data[:, :3] = (pred_data[:, :3] - og_mean)/og_std

        min_x_gt = np.min(gt_data[:, 0])
        min_x_pr = np.min(pred_data[:, 0])
        min_x_dif =  min_x_pr - min_x_gt

        min_y_dif =  np.min(pred_data[:, 1]) - np.min(gt_data[:, 1])

        pred_data[:, 0] -= min_x_dif
        pred_data[:, 1] -= min_y_dif

        gt_planes = extract_planes(gt_data)
        pred_planes = extract_planes(pred_data)
        #print("-----------------------------------------------------------")
        #print("sample name: ", name)
        #print("# of prediction planes ", len(pred_planes))
        #print("# of gt planes ", len(gt_planes))
        
        # get boundaries of each plane
        gt_plane_boundaries = get_boundaries(gt_planes)
        pred_plane_boundaries = get_boundaries(pred_planes)

        # match planes
        tmp_pred_plane_boundaries = pred_plane_boundaries.copy()
        tmp_gt_plane_boundaries = gt_plane_boundaries.copy()

        gt_matches, tp, fp, fn = get_match(gt_plane_boundaries, pred_plane_boundaries, threshold=0.4)

        pred_plane_boundaries = tmp_pred_plane_boundaries
        gt_plane_boundaries = tmp_gt_plane_boundaries

        pred_to_gt_match = {}

        for key, _ in pred_planes.items():
            for _, x in enumerate(gt_matches):
                if x[1] == key:
                    pred_to_gt_match[key] = x[0]
                    break
        
        if len(pred_to_gt_match):
            gt_sympy_planes = fit_plane_ransac(gt_planes, dense_std, dense_mean, debug)
            gt_tilts = calc_tilt(gt_sympy_planes)
            pred_sympy_planes = fit_plane_ransac(pred_planes, dense_std, dense_mean, debug)
            pred_tilts = calc_tilt(pred_sympy_planes)

            tilt_error = calc_tilt_error(pred_tilts, gt_tilts, pred_to_gt_match)
            if len(tilt_error):
                sum_absolute_tilt_errors = 0
                for key, err in tilt_error.items():
                    sum_absolute_tilt_errors += abs(err)
                average_plane_absolute_tilt_errors.append(sum_absolute_tilt_errors/len(tilt_error))

                plane_to_area_difference_percentage_over_gt_area = calc_area_difference_percentage_over_gt_area(gt_plane_boundaries, pred_plane_boundaries, pred_to_gt_match)
                sum_absolute_plane_area_diff = 0
                for key, diff in plane_to_area_difference_percentage_over_gt_area.items():
                    sum_absolute_plane_area_diff += abs(diff)
                average_absolute_plane_area_diff.append(sum_absolute_plane_area_diff/len(plane_to_area_difference_percentage_over_gt_area))    

        completeness = tp/(tp + fn)
        correctness = tp/(tp + fp)
        quality = tp/(tp+fn+fp)


        all_complt.append(completeness)
        all_corr.append(correctness)
        all_qual.append(quality)

        if debug:
            gt_data[:, :3] = (gt_data[:, :3] * dense_std) + dense_mean
            pred_data[:, :3] = (pred_data[:, :3] * og_std) + og_mean
            visualize_3d(gt_planes)
            visualize_3d(pred_planes, gt_matches)
            input('Press Enter key...')

    metrics = {"avg_completeness":float(sum(all_complt)/len(all_complt)),
               "avg_correctness":float(sum(all_corr)/len(all_corr)),
               "avg_quality":float(sum(all_qual)/len(all_qual)),
               "avg_abs_tilt_error":float(sum(average_plane_absolute_tilt_errors)/len(average_plane_absolute_tilt_errors))}

    with open(metric_saving_dir, "w") as outfile:
        json.dump(metrics, outfile)
