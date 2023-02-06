import os
import torch
import json
import sys
import numpy as np
sys.path.append("../")
import graphx
import utility
import pickle5 as pickle


def calc_psgn_loss(ground_truth_dir, results_dir, output_losses_dir, approach):
    output = {}
    #fill these in
    samples = [os.path.splitext(name)[0] for name in os.listdir(os.path.join(ground_truth_dir, 'annotation'))]
    gt = []
    pred = []

    # can be used for sat2lidar and psgn
    if approach == 'psgn' or approach=='sat2pc':
        results = torch.load(results_dir, map_location=torch.device('cuda'))
        for sample in samples:
            f = open(os.path.join(os.path.join(ground_truth_dir, 'annotation'), sample + '.json'))
            gt_lidar = np.asarray(json.load(f)['lidar'], 'float32')
            if gt_lidar.shape[0] == 3:
                gt_lidar = np.transpose(gt_lidar)
            gt_pc_mean = np.mean(gt_lidar, 0, keepdims=True)
            gt_pc_std = np.std(gt_lidar, 0, keepdims=True)
            gt_lidar -= gt_pc_mean
            gt_lidar /= gt_pc_std
            gt.append(torch.tensor(gt_lidar).cuda())
            
            f = False
            for res in results:
                if f :
                    break
                for _name, result in res.items():
                    if str(_name) == str(sample): 
                        pred.append(torch.tensor(result['final_out']))
                        f = True
                        break
    elif approach == 'graphx':
        for sample in samples:
            f = open(os.path.join(os.path.join(ground_truth_dir, 'annotation'), sample + '.json'))
            gt_lidar = np.asarray(json.load(f)['lidar'], 'float32')
            if gt_lidar.shape[0] == 3:
                gt_lidar = np.transpose(gt_lidar)
            gt_pc_mean = np.mean(gt_lidar, 0, keepdims=True)
            gt_pc_std = np.std(gt_lidar, 0, keepdims=True)
            gt_lidar -= gt_pc_mean
            gt_lidar /= gt_pc_std
            gt.append(torch.tensor(gt_lidar).cuda())


            with (open(os.path.join(results_dir, sample + ".out"), "rb")) as openfile:
                lidar_raw = pickle.load(openfile)[0]

            lidar = [[], [], []]
            num = 0
            for i in range(len(lidar_raw)):
                lidar[0].append(lidar_raw[i][0])  # X
                lidar[1].append(lidar_raw[i][1])  # Y
                lidar[2].append(lidar_raw[i][2])  # Z
                num = num + 1
            
            lidar = np.array(lidar).T
            pred.append(torch.tensor(lidar).cuda())

    #emd_grx = networks.get_emd([torch.tensor(neighbors).cuda()], [torch.tensor(gt_point).cuda()])
    #emd_res = networks.get_emd([torch.tensor(resnet_lidar).cuda()], [torch.tensor(gt_point).cuda()])
    #cham_grx = networks.get_chamfer([torch.tensor(neighbors)], [torch.tensor(gt_point)])
    #cham_res = networks.get_chamfer([torch.tensor(resnet_lidar)], [torch.tensor(gt_point)])


    chamfer_loss  = graphx.get_chamfer(pred, gt)
    emd_loss  = graphx.get_emd(pred, gt)

    output.update(final_chamfer_loss = chamfer_loss)
    output.update(final_emd_loss = emd_loss)

    pred = utility.remove_padding(pred)
    gt = utility.remove_padding(gt)

    no_pad_chamfer_loss  = graphx.get_chamfer(pred, gt)
    no_pad_emd_loss  = graphx.get_emd(pred, gt)

    outline_iou, _ = utility.get_outline_iou(pred, gt)

    output.update(final_no_pad_chamfer_loss = no_pad_chamfer_loss)
    output.update(final_no_pad_emd_loss = no_pad_emd_loss)
    output.update(outline_iou = outline_iou)

    losses_dict = {}
    for key, val in output.items():
        if torch.is_tensor(val):
            losses_dict[key] = val.cpu().item()
        else:
            losses_dict[key] = val

    with open(output_losses_dir, "w") as outfile:
        json.dump(losses_dict, outfile)

def calc_graphx_losses(ground_truth_dir, results_dir, output_losses_dir):
    asdd = 1