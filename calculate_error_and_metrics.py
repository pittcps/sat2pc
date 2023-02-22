import os
import json
import sys
sys.path.append("./PyTorchEMD/")
import evaluation
import data_util
import pptk
import utility
import torch
import numpy as np
import pickle5 as pickle
import shutil
import argparse


def run_performance_evaluation_on_sat2pc_result(dataset_dir, result_file_to_use, final_report_loc, segmented_gt_planes_dir, is_dense, debug):
    metric_saving_dir = ".\\evaluation\\logs\\tmp\\run1_metrics.json"

    segmentation_results_dir = '.\\evaluation\\logs\\tmp\\seq_res'

    prepared_prediction_data_dir = '.\\evaluation\\logs\\tmp\\pred_data'

    losses_dir = ".\\evaluation\\logs\\tmp\\loss_run1.json"

    averages = {"avg_completeness" : 0, "avg_correctness": 0, "avg_quality": 0, 'final_chamfer_loss': 0, 'final_emd_loss': 0, "final_no_pad_chamfer_loss": 0,
                "final_no_pad_emd_loss": 0, "outline_iou": 0, "avg_abs_tilt_error": 0}

    if os.path.exists(".\\evaluation\\logs\\tmp"):
        shutil.rmtree(".\\evaluation\\logs\\tmp")
    os.makedirs(".\\evaluation\\logs\\tmp")     

    evaluation.calc_planar_iou.calc_planar_metrics(dataset_dir, result_file_to_use, prepared_prediction_data_dir, segmentation_results_dir, \
        segmented_gt_planes_dir, metric_saving_dir, 'sat2pc', is_dense, debug)
    evaluation.calc_loss.calc_psgn_loss(dataset_dir, result_file_to_use, losses_dir, 'sat2pc')
    f = open(losses_dir)
    loss_data = json.load(f)

    for key in averages.keys():
        if key in loss_data.keys():
            averages[key] += loss_data[key]
        
    f = open(metric_saving_dir)
    metric_data = json.load(f)
    for key in averages.keys():
        if key in metric_data.keys():
            averages[key] += metric_data[key]

    with open(final_report_loc, "w") as outfile:
        json.dump(averages, outfile)
    print(averages)

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--data-dir", default="./datasets/extended_dataset/aggregated/splitted")
    parser.add_argument("--results", default="./results/test.res")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--final-output", default='./loss_and_metrics.json')
    parser.add_argument("--segmented-gt-planes", default= './datasets/extended_dataset/aggregated/splitted/ground_truth_segmentation_results/test')
    parser.add_argument("--is-dense", default=True)
    parser.add_argument("--debug", default=False)
    args = parser.parse_args()


    run_performance_evaluation_on_sat2pc_result(os.path.join(args.data_dir, args.dataset_split), \
                                                args.results, args.final_output, args.segmented_gt_planes, args.is_dense, args.debug)