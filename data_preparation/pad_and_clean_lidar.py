# load gt
# scale up the mask outlines
# for each point in og lidar:
#   if it's not in any of the masks ignore
#   if it's in both masks keep with same values
#   if it's in scaled one only, keep but change z to -9999

import numpy as np
import json
import os
import sys
import random
from shapely.geometry import Polygon, Point

ROOT_DIR = os.path.abspath("../")
sys.path.append('../')
import MercatorProjection
import data_util

def get_adjustment_offset(ds_id):
    if ds_id == 'FL_1_2':
        return [140, -90]
    elif ds_id == 'Austin':
        return [100, -90]
    elif ds_id == "Cleaveland":
        return [100, -130]
    elif ds_id == 'FL_3':
        return [60, -90]
    else:
        return [0, 0]

def filter_lidar(segments, lidar, wayid, zoom, w, h, srs, offset, scale, ds_id, plot, adjustment_offset):
    _, center_lat, center_lon = data_util.get_center_latlon_from_wayid(wayid)
    center_point = MercatorProjection.G_LatLng(center_lat, center_lon)
    segments = data_util.prepare_segments(segments)
    segments = data_util.convert_segments_to_lat_lon(segments, center_point, zoom, w, h, srs, offset, scale, adjustment_offset)
    polys = []

    for s in segments:
        polys.append(Polygon(s))

    new_lidar = [[], [], []]
    rgb = []

    for i in range(len(lidar[0])):
        lat = lidar[0][i]
        lon = lidar[1][i]

        p = Point(lat, lon)
        found = 0
        for j, poly in enumerate(polys):
            if poly.contains(p):
                if j%3 == 0:
                    rgb.append([float(0) / 255, float(255) / 255, float(0)/255])
                elif j%3 == 1:
                    rgb.append([float(255) / 255, float(0) / 255, float(0)/255])
                else:
                    rgb.append([float(0) / 255, float(0) / 255, float(255)/255])
                found = 1
                break
        if found == 1:
            for j in range(3):
                new_lidar[j].append(lidar[j][i])
    
    num = len(new_lidar[0])
    if plot == True:
        data_util.plot_3d(new_lidar, num, rgb)
    return np.asarray(new_lidar)


def clean_pad_lidar(segments, lidar, wayid, zoom, w, h, srs, offset, scale, ds_id, mean_height, median_height, adjustment_offset):
    print("DS ID: ", ds_id)
    pad_offset = 500
    height_threshold = 130 # for unscaled
    _, center_lat, center_lon = data_util.get_center_latlon_from_wayid(wayid)
    center_point = MercatorProjection.G_LatLng(center_lat, center_lon)
    segments = data_util.prepare_segments(segments)
    segments_og = data_util.convert_segments_to_lat_lon(segments, center_point, zoom, w, h, srs, offset, scale, adjustment_offset)
    poly = Polygon(segments_og[0])

    for s in segments_og:
        try:
            poly = poly.union(Polygon(s))
        except:
            print("handeling topology exepction")
            poly = poly.union(Polygon(s).buffer(0))

    poly_scaled = Polygon(poly.buffer(150.0).exterior)
    new_lidar = [[], [], []]
    rgb = []

    for i in range(len(lidar[0])):
        lat = lidar[0][i]
        lon = lidar[1][i]

        p = Point(lat, lon)
        found = 0
        
        if poly_scaled.contains(p):
            found = 1

        if found == 0:
            continue

        found = 0

        if poly.contains(p):
            found = 1

        if found == 1:
            
            if lidar[2][i] < median_height - height_threshold: #mean_height - height_threshold:
                for j in range(2):
                    new_lidar[j].append(lidar[j][i])
                new_lidar[2].append(median_height - pad_offset)
            else:

                for j in range(3):
                    new_lidar[j].append(lidar[j][i])
        else:
            for j in range(2):
                new_lidar[j].append(lidar[j][i])
            new_lidar[2].append(median_height - pad_offset)
    return np.asarray(new_lidar)

def subsample(lidar, n_points, plot):
    print("before subsample ", lidar.shape)
    if lidar.shape[1] > n_points:
        selected_idx = random.sample(range(0, lidar.shape[1]), n_points)
        lidar = lidar[:, selected_idx]
        if plot:
            print("subsampled shape: ", lidar.shape)
            data_util.plot_3d(lidar)
            input("...")
    
    return lidar

def pad_clean_subsample_lidars(input, output, plot, srs, zoom, width, height, number_of_points_to_subsample):
    print("Padding, cleaning, and subsampling lidars\n")
    os.makedirs(os.path.join(output, 'annotation_padded'), exist_ok=True)

    wayid_addr = os.path.join(input, 'wayids')
    lidar_addr = os.path.join(input, 'lidar')
    ann_dir = os.path.join(output, 'annotation')
    files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]

    for i, f in enumerate(files):
        if plot and i > 5:
            break
        print("sample: ", f)

        ann_out_addr = os.path.join(os.path.join(output, 'annotation_padded'), f)
        file_basename = os.path.splitext(f)[0]
        ann_path = os.path.join(ann_dir, f)
        fi = open(ann_path)
        ann = json.load(fi)
        fi.close()

        lidar, offset, scale = data_util.load_lidar(os.path.join(lidar_addr, file_basename + '.las'))
        wayid = data_util.load_json(os.path.join(wayid_addr,  f))
        segments = ann['points']
        ds_id = ann['dataset_id']
        adjustment_offset = get_adjustment_offset(ds_id)

        filtered_lidar = filter_lidar(segments, lidar, wayid, zoom, width, height, srs, offset, scale, ds_id, plot, adjustment_offset)

        if filtered_lidar.shape[0] == 3:
            filtered_lidar = filtered_lidar.T
        
        mean_height = np.mean(filtered_lidar[:, 2])
        median_height = np.median(filtered_lidar[:, 2]) 
        msks = np.array(ann['masks'])

        new_lidar = clean_pad_lidar(segments, lidar, wayid, zoom, width, height, srs, offset, scale, ds_id, mean_height, median_height, adjustment_offset)

        new_lidar = subsample(new_lidar, number_of_points_to_subsample, plot)

        if new_lidar.shape[0] == 0:
            print("skipping sample, no points in the final lidar!")
            continue

        if plot == True:
            print('median height: ', median_height)
            print('mean height: ', mean_height)
            data_util.plot_3d(new_lidar)

        if len(new_lidar[0]) == 0:
            print("zero points found!!")
            continue

        ann['lidar'] = new_lidar.tolist()

        with open(ann_out_addr, 'w') as outfile:
            json.dump(ann, outfile)
