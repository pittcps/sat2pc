import sys
sys.path.append("./PyTorchEMD/")
import glob
import os
import re
import time
import torch
import graphx
from sat2lidar_dataset import Sat2LidarDataset
from torch.utils.tensorboard import SummaryWriter
import neuralnet_pytorch.gin_nnt as gin
from graphx.networks import *
from sat2lidar_model import Sat2LidarNetwork
import engine

def get_graphx_model(args):
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
        net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 224, 224), (bs, n_points, 3), img_enc, pc_enc, pc_dec, adain=adain, projection=projection, weight_decay=None)

        mon.model_name = name
        mon.set_path('./graphx_weight/')
        states = mon.load('training.pt', method='torch', map_location='cpu')
        if states != None:
            net.load_state_dict(states['model_state_dict'])
        return net, n_points
    
    gin.parse_config_file(config_file)
    net, n_points = create_model()
    return net, n_points

def collate_fn(batch):
    return tuple(zip(*batch))

def log_loss_info(writer, mode, losses, epoch):
    for loss, val in losses.items():
        writer.add_scalar("log/Loss/{}_{}".format(mode, loss), val, epoch)

def main(args):

    batch_size = 4
    device = "cuda:"+args.gpu
    print("\ndevice: {}".format(device))
    alpha = torch.tensor(1).to(device)
    graphx_model, n_points = get_graphx_model(args)
    graphx_model = graphx_model.to(device)
    torch.cuda.empty_cache()
    model = Sat2LidarNetwork(graphx_model, n_points, device).to(device)

    print('Args: ', args)

    # ---------------------- prepare data loader ------------------------------- #
    
    _d_train = Sat2LidarDataset(image_dir = os.path.join(args.data_dir, os.path.join('train', 'image_filtered')), pc_num = n_points, ann_dir = os.path.join(args.data_dir, os.path.join('train', 'annotation')), mode='train')
    _d_val = Sat2LidarDataset(image_dir = os.path.join(args.data_dir, os.path.join('val', 'image_filtered')), pc_num = n_points, ann_dir = os.path.join(args.data_dir, os.path.join('val', 'annotation')), mode='val')
    
    d_train = torch.utils.data.DataLoader(_d_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    d_val = torch.utils.data.DataLoader(_d_val, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # -------------------------------------------------------------------------- #
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0) 
    milestones = [200, 800, 1300, 2000]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.3)
    
    start_epoch = 0
    best_eval_loss = 99999999
    prefix, ext = os.path.splitext(args.ckpt_path)

    ckpts = glob.glob(prefix + "_best" + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        torch.cuda.empty_cache()
        best_eval_loss = engine.evaluate(model, d_val, device, alpha, False, args)['total_loss']

    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
    writer = SummaryWriter()

    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
        best = False
            
        A = time.time()

        train_losses = engine.train_one_epoch(model, optimizer, d_train, device, epoch, alpha, args)

        A = time.time() - A

        if epoch > args.warmup_epochs:
            lr_scheduler.step()

        print('Train loss: ', train_losses)

        log_loss_info(writer, 'train', train_losses, epoch)
        
        B = time.time()
        eval_losses = engine.evaluate(model, d_val, device, alpha, False, args)
        B = time.time() - B

        if best_eval_loss > eval_losses['total_loss']:
            best_eval_loss = eval_losses['total_loss']
            best = True

        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
        print('Validation loss: ', eval_losses)

        log_loss_info(writer, 'val', eval_losses, epoch)

        engine.save_ckpt(model, optimizer, lr_scheduler, trained_epoch, args.ckpt_path, eval_info=str(eval_losses), best=best)
    
    # -------------------------------------------------------------------------- #
    writer.close()
    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--config", default = "./configs/graphx.gin")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="sat2lidar")
    parser.add_argument("--data-dir", default="./datasets/extended_dataset/aggregated/splitted")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results", default="./results/val.res")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.ckpt_path is None:
        args.ckpt_path = "./graphx_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
    
    main(args)
