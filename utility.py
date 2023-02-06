import sys
import torch

sys.path.append("./PyTorchEMD/")
from emd import earth_mover_distance

sys.path.append("../")
import data_util
from statistics import mean
from shapely.geometry import Polygon, Point
from polygonX import pgx
'''
def calc_emd(pred_pc, gt_pc, reduce="mean", normalized=False):
    import graphx
    loss = sum([graphx.normalized_chamfer_loss(pred[None], gt[None], reduce=reduce, normalized=normalized) for pred, gt in zip(pred_pc, gt_pc)]) / len(
            gt_pc) if isinstance(gt_pc, (list, tuple)) else graphx.normalized_chamfer_loss(pred_pc, gt_pc, reduce=reduce, normalized=normalized)

    return loss
'''

def remove_padding(lidar, threshold = 0.5):
    n_lidar = [] 

    for pc in lidar:
        #print("pc shape ", pc.shape)
        median_h = pc[:, 2].median()
        #print( median_h - (median_h*threshold))
        #print("median shape ", median_h.shape)
        idx = (pc[:, 2] >= median_h - (median_h*threshold)).nonzero(as_tuple=False)
        p = pc[idx].squeeze()
        #p = torch.gather(pc, dim = 0, index=idx)

        n_lidar.append(p)
    return n_lidar



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


'''
def calc_emd(pred, gt):
    import torch
    import numpy as np
    sys.path.append("./emd/")
    import emd_module as emd
    print("pred shape: ", pred.shape)
    print("gt shape: ", pred.shape)

    total_loss = []
    eps = 0.005
    iters = 50
    EMD = emd.emdModule()
    for b in range(pred.shape[0]):

        if pred[b].shape == gt[b].shape:
            # probably need to add batch
            dist = EMD(pred[b].unsqueeze(0), gt[b].unsqueeze(0), eps, iters)
            sample_loss = torch.sqrt(dist).mean(1)

        elif pred[b].shape[0] > gt[b].shape[0]:

            num_sub_sample_reps = int((pred[b].shape[0] + gt[b].shape[0] - 1) / gt[b].shape[0])

            sample_loss = torch.tensor(0.)
            for _ in range(num_sub_sample_reps):
                #resampled_idx = MDS_module.minimum_density_sample(pred[:, 0:3, :].transpose(1, 2).contiguous(), gt.shape[1], mean_mst_dis) 
                idx = np.random.choice(pred[b].shape[0], gt[b].shape[0], replace=False)
                #X = pred[b][idx][0]
                #Y = pred[b][idx][1]
                #Z = pred[b][idx][2]
                #pred_subsample = torch.stack((X, Y, Z), 1)
                pred_subsample = pred[b][idx].unsqueeze(0)
                print("pred_subsample shape", pred_subsample.shape)
                # probably need to add batch

                dist = EMD(pred_subsample, gt, eps, iters)
                emd_loss = torch.sqrt(dist).mean(1)
                sample_loss = sample_loss + emd_loss

            sample_loss =  sample_loss / torch.Tensor(num_sub_sample_reps)

        else:
            # ground truth shouldn't have more points
            print(pred[b].shape[0], " vs. " , gt[b].shape[0])
            assert(False)

        total_loss.append(sample_loss)

    return torch.Tensor(total_loss) 

    '''