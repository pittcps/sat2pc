import sys
sys.path.append("./PyTorchEMD/")
import glob
import os
import re
import torch
from sat2lidar_dataset import Sat2LidarDataset
import neuralnet_pytorch.gin_nnt as gin
from graphx.networks import *
from sat2lidar_model import Sat2LidarNetwork
import engine

def get_graphx_model(args, load_weights):
    config_file = args.config
    gin.external_configurable(CNN18Encoder, 'cnn18_enc')
    gin.external_configurable(PointCloudEncoder, 'pc_enc')
    gin.external_configurable(PointCloudDecoder, 'pc_dec')
    gin.external_configurable(PointCloudResDecoder, 'pc_resdec')
    gin.external_configurable(PointCloudResGraphXDecoder, 'pc_resgraphxdec')
    gin.external_configurable(PointCloudResGraphXUpDecoder, 'pc_upresgraphxdec')
    gin.external_configurable(PointCloudResLowRankGraphXUpDecoder, 'pc_upreslowrankgraphxdec')

    @gin.configurable('GraphX')
    def create_model(name, img_enc, pc_enc, pc_dec, adain=True, projection=True, decimation=None, color_img=True, n_points=250, bs=4, checkpoint_folder=None):
        if decimation is not None:
            pc_dec = partial(pc_dec, decimation=decimation)
        print("Number of points = ", n_points)
        net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 256, 256), (bs, n_points, 3), img_enc, pc_enc, pc_dec, adain=adain, projection=projection, weight_decay=None)

        return net, n_points, name
    
    gin.parse_config_file(config_file)
    net, n_points, name = create_model()

    if load_weights:
        mon.model_name = name
        mon.set_path('./graphx_weight/')
        states = mon.load('training.pt', method='torch', map_location='cpu')
        net.load_state_dict(states['model_state_dict'])

    return net, n_points
    
def collate_fn(batch):
    return tuple(zip(*batch))

def main(args):
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\ndevice: {}".format(device))

    load_weights = False
    alpha = torch.tensor(1).to(device)

    graphx_model, n_points = get_graphx_model(args, load_weights)
    graphx_model = graphx_model.to(device)
    model = Sat2LidarNetwork(graphx_model, n_points, device).to(device)
        
    # ---------------------- prepare data loader ------------------------------- #
    _d_test = Sat2LidarDataset(image_dir = os.path.join(args.data_dir, os.path.join(args.dataset_split, 'image_filtered')), pc_num = n_points, ann_dir = os.path.join(args.data_dir, os.path.join(args.dataset_split, 'annotation')), mode=args.dataset_split, augment=False)
    d_test = torch.utils.data.DataLoader(_d_test, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    
    # -------------------------------------------------------------------------- #

    print('Args: ', args)

    start_epoch = 0

    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path, map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print('No checkpoints')
    
    print("\nTesting on epoch {}".format(start_epoch))
    
    # ------------------------------- test ------------------------------------ #

    losses = engine.evaluate(model, d_test, device, alpha, True, args)

    print(losses)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "./configs/sat2pc.gin")

    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="sat2lidar")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--data-dir", default=".\datasets\extended_dataset\\aggregated\\splitted")
    
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results", default="./results/test.res")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--iters", type=int, default=200, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()

    if args.dataset_split != 'test':
        args.results="./results/" + args.dataset_split + ".res"
    
    if args.ckpt_path is None:
        args.ckpt_path = "./graphx_{}.pth".format(args.dataset)
    
    main(args)
    
    
