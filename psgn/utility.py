import torch
import sys
sys.path.append("./PyTorchEMD/")
from emd import earth_mover_distance
import os
current_d = os.path.dirname(os.path.realpath(__file__))
parent_d = os.path.dirname(current_d)
sys.path.append(parent_d)
import neuralnet_pytorch as nnt
import data_util
from statistics import mean


def remove_padding(lidar, threshold = 0.5):
    n_lidar = [] 

    for pc in lidar:
        print("pc shape ", pc.shape)
        median_h = pc[:, 2].median()
        #print( median_h - (median_h*threshold))
        #print("median shape ", median_h.shape)
        idx = (pc[:, 2] >= median_h - (median_h*threshold)).nonzero(as_tuple=False)
        p = pc[idx].squeeze()
        #p = torch.gather(pc, dim = 0, index=idx)

        n_lidar.append(p)
    return n_lidar

def normalized_chamfer_loss(pred, gt, reduce='mean', normalized=False):
    if normalized:
        max_dist, _ = torch.max(torch.sqrt(torch.sum(gt ** 2., -1)), -1)
        origin = torch.mean(gt, -2)
        pred = (pred - origin) / max_dist
        gt = (gt - origin) / max_dist

    loss = nnt.chamfer_loss(pred, gt, reduce=reduce)

    #alpha = 0.01
    #min_dists, _ = T.min(T.cdist(pred, gt), dim=2)
    #regularization = T.max(min_dists)
    #return loss + alpha * regularization if reduce == 'sum' else (loss + alpha * regularization) * 1. 
    return loss if reduce == 'sum' else loss * 1. 


def calc_chamfer(pred_pc, gt_pc, reduce="mean"):
    loss = sum([nnt.chamfer_loss(pred[None], gt[None], reduce=reduce) for pred, gt in zip(pred_pc, gt_pc)]) / len(
            gt_pc) if isinstance(gt_pc, (list, tuple)) else nnt.chamfer_loss(pred_pc, gt_pc, reduce=reduce)
    return loss


def calc_emd(pred_pc, gt_pc, transpose=False):
    if isinstance(gt_pc, (list, tuple)): 
        loss = sum([earth_mover_distance(pred[None], gt[None], transpose=transpose)/pred.shape[0] for pred, gt in zip(pred_pc, gt_pc)]) / len(gt_pc)
    else:
        loss = earth_mover_distance(pred_pc, gt_pc, transpose=transpose)/pred_pc.shape[1]

    return loss

def get_outline_iou(pred_pc, gt_pc):

    l = 5.5
    ious = []
    
    for pc, gt in zip(pred_pc, gt_pc):
        #pred_2d -= torch.mean(pred_2d, 0, True)
        #pred_2d /= torch.std(pred_2d, 0, True)

        pred_vertx = data_util.get_pointcolud_outline(pc.cpu(), l = l)
        gt_vertx = data_util.get_pointcolud_outline(gt.cpu(), l = l)

        intersect = pred_vertx.intersection(gt_vertx).area
        union = pred_vertx.union(gt_vertx).area
        iou = intersect / union
        ious.append(iou)

    
        '''

        poly_pred = Polygon(pp)
        poly_gt = Polygon(gp)

        intersect = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area
        iou = intersect / union
        ious.append(iou)

        '''

    return mean(ious), ious

