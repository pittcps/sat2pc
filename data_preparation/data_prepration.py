import os
import sys
import argparse
import extract_annotations
import image_filtration
import add_ds_info
import pad_and_clean_lidar
ROOT_DIR = os.path.abspath("../")
sys.path.append('../')
import data_util

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='..\datasets\\extended_dataset\\unprocessed\\FL_3')
parser.add_argument('--output', type=str, default='..\datasets\\extended_dataset\\processed\\FL_3')
parser.add_argument('--use_wayid', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--number_subsample_points', type=int, default=3000)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

dataset_info = data_util.load_json(os.path.join(args.input, 'dataset_info.json'))

extract_annotations.extract_annotations(args.input, args.output)
add_ds_info.add_ds_info(args.output, dataset_info['dataset_id'], dataset_info['lidar_srs'], dataset_info['img_zoom'], dataset_info['img_width'],\
    dataset_info['img_height'])
image_filtration.filter_all_images(args.input, args.output, args.use_wayid, args.debug, dataset_info['lidar_srs'], dataset_info['img_zoom'], \
    dataset_info['img_width'], dataset_info['img_height'])
pad_and_clean_lidar.pad_clean_subsample_lidars(args.input, args.output, args.debug, dataset_info['lidar_srs'], dataset_info['img_zoom'], \
    dataset_info['img_width'], dataset_info['img_height'], args.number_subsample_points)
