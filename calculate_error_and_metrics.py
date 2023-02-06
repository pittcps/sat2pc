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


def get_prediction(sample_name, method):
    if method == "sat2pc":
        res_dir = '.\\evaluation\\logs\\results\\performance_results\\sat2pc_test_results\\test_result_run2.res'
        results = torch.load(res_dir, map_location=torch.device('cpu'))
        f = False
        for res in results:
            if f :
                break
            for _name, result in res.items():
                if str(_name) == str(sample_name): 
                    lidar = np.asarray(result['final_out'])
                    f = True
                    break
        return lidar
    elif method == "psgn":
        res_dir = '.\\evaluation\\logs\\results\\performance_results\\psgn_test_results\\PSGN_test.res'
        results = torch.load(res_dir, map_location=torch.device('cpu'))
        f = False
        for res in results:
            if f :
                break
            for _name, result in res.items():
                if str(_name) == str(sample_name): 
                    lidar = np.asarray(result['final_out'])
                    f = True
                    break
        return lidar
    elif method == "graphx":
        res_dir = os.path.join('.\\evaluation\\logs\\results\\performance_results\\graphx_chamfer_test_results\\graphx_test_results_1', sample_name + ".out")
        with (open(res_dir, "rb")) as openfile:
            raw_data = pickle.load(openfile)
        raw_data = raw_data[0]

        data = [[], [], []]
        for i in range(len(raw_data)):
            data[0].append(raw_data[i][0])  # X
            data[1].append(raw_data[i][1])  # Y
            data[2].append(raw_data[i][2])  # Z
        return data
    else:
        print("method not found, please enter a valid method")
    
def plot_3_density(resnet_lidar, neighbors, gt_point, r, gt_pc_std, gt_pc_mean):
    #D1: density of sat2pc c+e
    #D2: density of chamfer graphx
    #D3: density of ground truth
    D1, D2, D3 = data_util.get_density(resnet_lidar, neighbors, gt_point, r)
    
    neighbors = (neighbors * gt_pc_std) + gt_pc_mean
    resnet_lidar = (resnet_lidar * gt_pc_std) + gt_pc_mean
    gt_point = (gt_point * gt_pc_std) + gt_pc_mean

    scale = [min(D3), max(D3)]
    print(scale)
    v1 = pptk.viewer(gt_point, D1)
    v1.color_map('jet', scale=scale)
    v1.set(point_size=5)
    v1.set(bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

    input('Currently seeing density of Sat2PC predictions, please press enter to continue ...')

    v2 = pptk.viewer(gt_point, D2)
    v2.color_map('jet', scale=scale)
    v2.set(point_size=5)
    v2.set(bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)
    input('Currently seeing density of GraphX predictions, please press enter to continue ...')

    v3 = pptk.viewer(gt_point, D3)
    v3.color_map('jet', scale=scale)
    v3.set(point_size=5)
    v3.set(bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

    input('Currently seeing density of ground truth, please press enter to plot the density box chart ...')
    v1.close()
    v2.close()
    v3.close()

def run_performance_evaluation(config_file, weights_to_run, experiment, final_report_loc):
    dataset_dir = ".\\datasets\\test\\"
    segmented_gt_planes_dir = '.\\evaluation\\ground_truth_segmentation_results\\test'
    debug = False

    weights_dir = [".\\saved_models\\{}\\run1\\graphx_sat2lidar-2000.pth".format(weights_to_run),
                ".\\saved_models\\{}\\run2\\graphx_sat2lidar-2000.pth".format(weights_to_run),
                ".\\saved_models\\{}\\run3\\graphx_sat2lidar-2000.pth".format(weights_to_run),
                ".\\saved_models\\{}\\run4\\graphx_sat2lidar-2000.pth".format(weights_to_run),
                ".\\saved_models\\{}\\run5\\graphx_sat2lidar-2000.pth".format(weights_to_run)]

    metric_saving_dir = [".\\evaluation\\logs\\metric_results\\{}\\sat2pc_test_results\\run1_metrics.json".format(experiment),
                        ".\\evaluation\\logs\\metric_results\\{}\\sat2pc_test_results\\run2_metrics.json".format(experiment),
                        ".\\evaluation\\logs\\metric_results\\{}\\sat2pc_test_results\\run3_metrics.json".format(experiment),
                        ".\\evaluation\\logs\\metric_results\\{}\\sat2pc_test_results\\run4_metrics.json".format(experiment),
                        ".\\evaluation\\logs\\metric_results\\{}\\sat2pc_test_results\\run5_metrics.json".format(experiment)]

    segmentation_results_dir = ['.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run1_segmentation_results'.format(experiment),
                                '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run2_segmentation_results'.format(experiment),
                                '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run3_segmentation_results'.format(experiment),
                                '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run4_segmentation_results'.format(experiment),
                                '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run5_segmentation_results'.format(experiment)]

    prepared_prediction_data_dir = ['.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run1_prediction_segmentation_data\\'.format(experiment),
                                    '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run2_prediction_segmentation_data\\'.format(experiment),
                                    '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run3_prediction_segmentation_data\\'.format(experiment),
                                    '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run4_prediction_segmentation_data\\'.format(experiment),
                                    '.\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\run5_prediction_segmentation_data\\'.format(experiment)]

    results_dir = [".\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\test_result_run1.res".format(experiment),
                ".\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\test_result_run2.res".format(experiment),
                ".\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\test_result_run3.res".format(experiment),
                ".\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\test_result_run4.res".format(experiment),
                ".\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\test_result_run5.res".format(experiment)]

    losses_dir = [".\\evaluation\\logs\\losses\\{}\\sat2pc_test_results\\loss_run1.json".format(experiment),
                ".\\evaluation\\logs\\losses\\{}\\sat2pc_test_results\\loss_run2.json".format(experiment),
                ".\\evaluation\\logs\\losses\\{}\\sat2pc_test_results\\loss_run3.json".format(experiment),
                ".\\evaluation\\logs\\losses\\{}\\sat2pc_test_results\\loss_run4.json".format(experiment),
                ".\\evaluation\\logs\\losses\\{}\\sat2pc_test_results\\loss_run5.json".format(experiment)]

    averages = {"avg_completeness" : 0, "avg_correctness": 0, "avg_quality": 0, 'final_chamfer_loss': 0, 'final_emd_loss': 0, "final_no_pad_chamfer_loss": 0,
                "final_no_pad_emd_loss": 0, "outline_iou": 0, "avg_abs_tilt_error": 0}

          
    os.makedirs(".\\evaluation\\logs\\metric_results\\{}\\sat2pc_test_results".format(experiment), exist_ok=True)     
    os.makedirs(".\\evaluation\\logs\\results\\{}\\sat2pc_test_results\\".format(experiment), exist_ok=True)
    os.makedirs(".\\evaluation\\logs\\losses\\{}\\sat2pc_test_results\\".format(experiment), exist_ok=True)
    
    number_of_runs = 0

    for i in range(5):
        if weights_to_run == "emd+chamfer" and i > 0:
            break

        os.makedirs(segmentation_results_dir[i], exist_ok=True)    
        os.makedirs(prepared_prediction_data_dir[i], exist_ok=True)  
        
        number_of_runs += 1

        os.system('python ./test.py --config ./configs/{} --ckpt-path {} --results {} --loss-log-loc {}'.format(config_file, weights_dir[i], results_dir[i], losses_dir[i]))
        evaluation.calc_planar_iou.calc_planar_metrics(dataset_dir, results_dir[i], prepared_prediction_data_dir[i], segmentation_results_dir[i], \
            segmented_gt_planes_dir, metric_saving_dir[i], 'sat2pc', debug)
        
        f = open(losses_dir[i])
        loss_data = json.load(f)

        for key in averages.keys():
            if key in loss_data.keys():
                averages[key] += loss_data[key]
        
        f = open(metric_saving_dir[i])
        metric_data = json.load(f)
        for key in averages.keys():
            if key in metric_data.keys():
                averages[key] += metric_data[key]

    for key in averages.keys():
        averages[key] /= number_of_runs

    with open(final_report_loc, "w") as outfile:
        json.dump(averages, outfile)
    print(averages)

def run_performance_evaluation_on_psgn(final_report_loc):
    dataset_dir = ".\\datasets\\test\\"
    segmented_gt_planes_dir = '.\\evaluation\\ground_truth_segmentation_results\\test'
    debug = False
    weights_dir = ".\\saved_models\\PointSet\\psgn-2000.pth"

    metric_saving_dir = ".\\evaluation\\logs\\metric_results\\performance_results\\psgn_test_results\\psgn_metrics.json"

    segmentation_results_dir = '.\\evaluation\\logs\\results\\performance_results\\psgn_test_results\\PSGN_segmentation_results'

    prepared_prediction_data_dir = '.\\evaluation\\logs\\results\\performance_results\\psgn_test_results\\PSGN_prediction_segmentation_data\\'

    results_dir = ".\\evaluation\\logs\\results\\performance_results\\psgn_test_results\\PSGN_test.res"

    losses_dir = ".\\evaluation\\logs\\losses\\performance_results\\psgn_test_results\\psgn_loss.json"

    averages = {"avg_completeness" : 0, "avg_correctness": 0, "avg_quality": 0, 'final_chamfer_loss': 0, 'final_emd_loss': 0, "final_no_pad_chamfer_loss": 0,
                "final_no_pad_emd_loss": 0, "outline_iou": 0, "avg_abs_tilt_error": 0}

    os.makedirs(".\\evaluation\\logs\\metric_results\\performance_results\\psgn_test_results", exist_ok=True)
    os.makedirs(".\\evaluation\\logs\\results\\performance_results\\psgn_test_results", exist_ok=True)
    os.makedirs(".\\evaluation\\logs\\losses\\performance_results\\psgn_test_results", exist_ok=True)
    
    os.makedirs(segmentation_results_dir, exist_ok=True)    
    os.makedirs(prepared_prediction_data_dir, exist_ok=True)  

    os.system('python .\\psgn\\test.py --ckpt-path {} --results {}'.format(weights_dir, results_dir))

    evaluation.calc_planar_iou.calc_planar_metrics(dataset_dir, results_dir, prepared_prediction_data_dir, segmentation_results_dir, \
        segmented_gt_planes_dir, metric_saving_dir, 'psgn', debug)
    
    evaluation.calc_loss.calc_psgn_loss(dataset_dir, results_dir, losses_dir, 'psgn')

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

def run_performance_evaluation_on_graphx(loss_func, final_report_loc):
    dataset_dir = ".\\datasets\\test\\"
    segmented_gt_planes_dir = '.\\evaluation\\ground_truth_segmentation_results\\test'
    debug = False

    weights_dir = [".\\saved_models\\graphx\\{}\\run1\\".format(loss_func),
                ".\\saved_models\\graphx\\{}\\run2\\".format(loss_func),
                ".\\saved_models\\graphx\\{}\\run3\\".format(loss_func),
                ".\\saved_models\\graphx\\{}\\run4\\".format(loss_func),
                ".\\saved_models\\graphx\\{}\\run5\\".format(loss_func)]

    metric_saving_dir = [".\\evaluation\\logs\\metric_results\\performance_results\\graphx_{}_test_results\\run1_metrics.json".format(loss_func),
                        ".\\evaluation\\logs\\metric_results\\performance_results\\graphx_{}_test_results\\run2_metrics.json".format(loss_func),
                        ".\\evaluation\\logs\\metric_results\\performance_results\\graphx_{}_test_results\\run3_metrics.json".format(loss_func),
                        ".\\evaluation\\logs\\metric_results\\performance_results\\graphx_{}_test_results\\run4_metrics.json".format(loss_func),
                        ".\\evaluation\\logs\\metric_results\\performance_results\\graphx_{}_test_results\\run5_metrics.json".format(loss_func)]

    segmentation_results_dir = ['.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run1_segmentation_results'.format(loss_func),
                                '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run2_segmentation_results'.format(loss_func),
                                '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run3_segmentation_results'.format(loss_func),
                                '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run4_segmentation_results'.format(loss_func),
                                '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run5_segmentation_results'.format(loss_func)]

    prepared_prediction_data_dir = ['.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run1_prediction_segmentation_data\\'.format(loss_func),
                                    '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run2_prediction_segmentation_data\\'.format(loss_func),
                                    '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run3_prediction_segmentation_data\\'.format(loss_func),
                                    '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run4_prediction_segmentation_data\\'.format(loss_func),
                                    '.\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\run5_prediction_segmentation_data\\'.format(loss_func)]

    results_dir = [".\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\graphx_test_results_1".format(loss_func),
                ".\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\graphx_test_results_2".format(loss_func),
                ".\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\graphx_test_results_3".format(loss_func),
                ".\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\graphx_test_results_4".format(loss_func),
                ".\\evaluation\\logs\\results\\performance_results\\graphx_{}_test_results\\graphx_test_results_5".format(loss_func)]

    losses_dir = [".\\evaluation\\logs\\losses\\performance_results\\graphx_{}_test_results\\loss_run1.json".format(loss_func),
                ".\\evaluation\\logs\\losses\\performance_results\\graphx_{}_test_results\\loss_run2.json".format(loss_func),
                ".\\evaluation\\logs\\losses\\performance_results\\graphx_{}_test_results\\loss_run3.json".format(loss_func),
                ".\\evaluation\\logs\\losses\\performance_results\\graphx_{}_test_results\\loss_run4.json".format(loss_func),
                ".\\evaluation\\logs\\losses\\performance_results\\graphx_{}_test_results\\loss_run5.json".format(loss_func)]

    averages = {"avg_completeness" : 0, "avg_correctness": 0, "avg_quality": 0, 'final_chamfer_loss': 0, 'final_emd_loss': 0, "final_no_pad_chamfer_loss": 0,
                "final_no_pad_emd_loss": 0, "outline_iou": 0, "avg_abs_tilt_error": 0}

    os.makedirs(".\\evaluation\\logs\\metric_results\\performance_results\\graphx_{}_test_results".format(loss_func), exist_ok=True)     
    os.makedirs(".\\evaluation\\logs\\results\\performance_results", exist_ok=True)
    os.makedirs(".\\evaluation\\logs\\losses\\performance_results\\graphx_{}_test_results\\".format(loss_func), exist_ok=True)
    
    number_of_runs = 0

    for i in range(5):

        os.makedirs(segmentation_results_dir[i], exist_ok=True)    
        os.makedirs(prepared_prediction_data_dir[i], exist_ok=True)  
        
        number_of_runs += 1

        os.system('python -m graphx.test --config ./configs/graphx.gin --ckpt-path {} --results {}'.format(weights_dir[i], results_dir[i]))
        evaluation.calc_planar_iou.calc_planar_metrics(dataset_dir, results_dir[i], prepared_prediction_data_dir[i], segmentation_results_dir[i], segmented_gt_planes_dir, metric_saving_dir[i], 'graphx', debug)
        evaluation.calc_loss.calc_psgn_loss(dataset_dir, results_dir[i], losses_dir[i], 'graphx')
        f = open(losses_dir[i])
        loss_data = json.load(f)

        for key in averages.keys():
            if key in loss_data.keys():
                averages[key] += loss_data[key]
        
        f = open(metric_saving_dir[i])
        metric_data = json.load(f)
        for key in averages.keys():
            if key in metric_data.keys():
                averages[key] += metric_data[key]

    for key in averages.keys():
        averages[key] /= number_of_runs

    with open(final_report_loc, "w") as outfile:
        json.dump(averages, outfile)
    print(averages)

def run_point_size_effect_evaluation():
    print("Running evaluation for 4000 points ...")
    run_performance_evaluation("sat2pc_4000.gin", "runs_4000_ps", "point_cloud_size_results", 
                               ".\\evaluation\\logs\\results\\point_cloud_size_results\\final_average_report_4000ps.json")
    print("Running evaluation for 6000 points ...")
    run_performance_evaluation("sat2pc_6000.gin", "runs_6000_ps", "point_cloud_size_results", 
                               ".\\evaluation\\logs\\results\\point_cloud_size_results\\final_average_report_6000ps.json")
    print("Running evaluation for 8000 points ...")
    run_performance_evaluation("sat2pc_8000.gin", "runs_8000_ps", "point_cloud_size_results", 
                               ".\\evaluation\\logs\\results\\point_cloud_size_results\\final_average_report_8000ps.json")
    print("Running evaluation for 3000 points ...")
    run_performance_evaluation("sat2pc.gin", "runs_3000_ps", "point_cloud_size_results", 
                               ".\\evaluation\\logs\\results\\point_cloud_size_results\\final_average_report_3000ps.json")
    print("**************************************\n")
    print("Reports can be found at .\\evaluation\\logs\\results\\point_cloud_size_results\\")
    print("final_average_report_3000ps.json")
    print("final_average_report_4000ps.json")
    print("final_average_report_6000ps.json")
    print("final_average_report_8000ps.json")
    
def run_alpha_effect_evaluation():
    print("Running evaluation for alpha = 0.2 ...")
    run_performance_evaluation("sat2pc.gin", "alpha_02_3000", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_0-2.json")
    print("Running evaluation for alpha = 0.3 ...")
    run_performance_evaluation("sat2pc.gin", "alpha_03_3000", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_0-3.json")
    print("Running evaluation for alpha = 0.5 ...")
    run_performance_evaluation("sat2pc.gin", "alpha_05_3000", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_0-5.json")
    print("Running evaluation for alpha = 0.8 ...")
    run_performance_evaluation("sat2pc.gin", "alpha_08_3000", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_0-8.json")
    print("Running evaluation for alpha = 1 ...")
    run_performance_evaluation("sat2pc.gin", "runs_3000_ps", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_1-0.json")
    print("Running evaluation for alpha = 1.5 ...")
    run_performance_evaluation("sat2pc.gin", "alpha_15_3000", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_1-5.json")
    print("Running evaluation for alpha = 2 ...")
    run_performance_evaluation("sat2pc.gin", "alpha_20_3000", "alpha_results", 
                               ".\\evaluation\\logs\\results\\alpha_results\\final_average_report_alpha_2-0.json")
    print("**************************************\n")
    print("Reports can be found at .\\evaluation\\logs\\results\\alpha_results\\")
    print("final_average_report_alpha_0-2.json")
    print("final_average_report_alpha_0-3.json")
    print("final_average_report_alpha_0-5.json")
    print("final_average_report_alpha_0-8.json")
    print("final_average_report_alpha_1-0.json")
    print("final_average_report_alpha_1-5.json")
    print("final_average_report_alpha_2-0.json")

def run_loss_effect_evaluation():
    print("Running evaluation for Chamfer + Chamfer ...")
    run_performance_evaluation("sat2pc.gin", "chamfer+chamfer", "loss_results", 
                               ".\\evaluation\\logs\\results\\loss_results\\final_average_report_chamfer+chamfer.json")
    print("Running evaluation for Chamfer + EMD ...")
    run_performance_evaluation("sat2pc.gin", "runs_3000_ps", "loss_results", 
                               ".\\evaluation\\logs\\results\\loss_results\\final_average_report_chamfer+emd.json")
    print("Running evaluation for EMD + Chamfer ...")
    run_performance_evaluation("sat2pc.gin", "emd+chamfer", "loss_results", 
                               ".\\evaluation\\logs\\results\\loss_results\\final_average_report_emd+chamfer.json")
    print("Running evaluation for EMD + EMD ...")
    run_performance_evaluation("sat2pc.gin", "emd+emd", "loss_results", 
                               ".\\evaluation\\logs\\results\\loss_results\\final_average_report_emd+emd.json")
    print("**************************************\n")
    print("Reports can be found at .\\evaluation\\logs\\results\\loss_results\\")
    print("final_average_report_chamfer+emd.json")
    print("final_average_report_chamfer+chamfer.json")
    print("final_average_report_emd+chamfer.json")
    print("final_average_report_emd+emd.json")

def run_qualitative_loss():
    dataset_dir = ".\\datasets\\test\\"
    samples = ["523638689", "524203109", "526245597", "526245688"]
    for sample in samples:
        img = data_util.load_image(os.path.join(dataset_dir, os.path.join("image_filtered", sample + ".png")))
        img.show()

        gt_point_cloud = np.asarray(data_util.load_json(os.path.join(dataset_dir, os.path.join("annotation", sample + ".json")))['lidar']).T

        gt_pc_mean = np.mean(gt_point_cloud, 0, keepdims=True)
        gt_pc_std = np.std(gt_point_cloud, 0, keepdims=True)

        gt_point_cloud -= gt_pc_mean
        gt_point_cloud /= gt_pc_std

        gt_point_cloud = utility.remove_padding([torch.tensor(gt_point_cloud).cpu()])[0]

        gt_point_cloud *= gt_pc_std
        gt_point_cloud += gt_pc_mean

        scale = [min(gt_point_cloud[:, 2]), max(gt_point_cloud[:, 2])]


        v1 = pptk.viewer(gt_point_cloud, gt_point_cloud[:, 2], title="Sample {}: Ground Truth".format(sample))
        v1.color_map('jet', scale = scale)
        v1.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)


        sat2pc_point_cloud = np.asarray(get_prediction(sample, "sat2pc"))

        sat2pc_point_cloud = utility.remove_padding([torch.tensor(sat2pc_point_cloud).cpu()])[0]

        sat2pc_point_cloud *= gt_pc_std
        sat2pc_point_cloud += gt_pc_mean
        input("Last window shows the ground truth for the sample, please press enter to visualize Sat2PC's predicted point cloud ...")

        v2 = pptk.viewer(sat2pc_point_cloud, sat2pc_point_cloud[:, 2], title="Sample {}: Sat2PC Prediction".format(sample))
        v2.color_map('jet', scale = scale)
        v2.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

        psgn_point_cloud = np.asarray(get_prediction(sample, "psgn"))

        psgn_point_cloud = utility.remove_padding([torch.tensor(psgn_point_cloud).cpu()])[0]

        psgn_point_cloud *= gt_pc_std
        psgn_point_cloud += gt_pc_mean
        input("Last window shows the sat2pc prediction for the sample, please press enter to visualize PointSet's predicted point cloud ...")

        v3 = pptk.viewer(psgn_point_cloud, psgn_point_cloud[:, 2], title="Sample {}: PointSet Prediction".format(sample))
        v3.color_map('jet', scale = scale)
        v3.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

        graphx_point_cloud = np.asarray(get_prediction(sample, "graphx")).T

        graphx_point_cloud = utility.remove_padding([torch.tensor(graphx_point_cloud).cpu()])[0]
        graphx_point_cloud *= gt_pc_std
        graphx_point_cloud += gt_pc_mean
        input("Last window shows the PointSet's prediction for the sample, please press enter to visualize GraphX's predicted point cloud ...")

        v4 = pptk.viewer(graphx_point_cloud, graphx_point_cloud[:, 2], title="Sample {}: GraphX Prediction".format(sample))
        v4.color_map('jet', scale = scale)
        v4.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)
        
        input("Please press enter to visualize the next sample ...")
        img.close()
        v1.close()
        v2.close()
        v3.close()
        v4.close()

def run_density_analysis():
    dataset_dir = ".\\datasets\\test\\"
    samples = ["524203109", "524203116", "524203233"]
    r = 0.5
    for sample in samples:
        gt_point_cloud = np.asarray(data_util.load_json(os.path.join(dataset_dir, os.path.join("annotation", sample + ".json")))['lidar']).T

        gt_pc_mean = np.mean(gt_point_cloud, 0, keepdims=True)
        gt_pc_std = np.std(gt_point_cloud, 0, keepdims=True)

        gt_point_cloud -= gt_pc_mean
        gt_point_cloud /= gt_pc_std
        gt_point_cloud = utility.remove_padding([torch.tensor(gt_point_cloud).cpu()])[0]

        graphx_point_cloud = np.asarray(get_prediction(sample, "graphx")).T
        graphx_point_cloud = utility.remove_padding([torch.tensor(graphx_point_cloud).cpu()])[0]

        sat2pc_point_cloud = np.asarray(get_prediction(sample, "sat2pc"))
        sat2pc_point_cloud = utility.remove_padding([torch.tensor(sat2pc_point_cloud).cpu()])[0]

        plot_3_density(sat2pc_point_cloud, graphx_point_cloud, gt_point_cloud, r, gt_pc_std, gt_pc_mean)

        data_util.plot_box_chart_density_dist(sat2pc_point_cloud, graphx_point_cloud, gt_point_cloud, r, 'Density Distribution', 'Sat2PC', 'GraphX-Conv')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default = "performance", choices=["performance", "qualitative", "density", "point_size", "alpha", "loss"])
    args = parser.parse_args()

    if args.eval == "performance":
        #run_performance_evaluation("sat2pc.gin", "runs_3000_ps", "performance_results", ".\\evaluation\\logs\\results\\performance_results\\sat2pc_final_average_report.json")
        #run_performance_evaluation_on_psgn(".\\evaluation\\logs\\results\\performance_results\\psgn_final_average_report.json")
        #run_performance_evaluation_on_graphx("chamfer", ".\\evaluation\\logs\\results\\performance_results\\graphx_chamfer_final_average_report.json")
        run_performance_evaluation_on_graphx("emd", ".\\evaluation\\logs\\results\\performance_results\\graphx_emd_final_average_report.json")
        print("Reports can be found at .\\evaluation\\logs\\results\\performance_results\\")
        print("sat2pc_final_average_report.json")
        print("psgn_final_average_report.json")
        print("graphx_chamfer_final_average_report.json")
        print("graphx_emd_final_average_report.json")
    elif args.eval == "qualitative":
        run_qualitative_loss()
    elif args.eval == "density":
        run_density_analysis()
    elif args.eval == "point_size":
        run_point_size_effect_evaluation()
    elif args.eval == "alpha":    
        run_alpha_effect_evaluation()
    elif args.eval == "loss":    
        run_loss_effect_evaluation()
