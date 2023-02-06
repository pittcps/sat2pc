import glob
import os
import re
import time
import torch
from sat2lidar_dataset import Sat2LidarDataset
from torch.utils.tensorboard import SummaryWriter
import engine
from psgn import PSGN

def collate_fn(batch):
    return tuple(zip(*batch))

def log_loss_info(writer, mode, losses, epoch):
    for loss, val in losses.items():
        writer.add_scalar("log/Loss/{}_{}".format(mode, loss), val, epoch)

def main(args):
    #todo: add config file support
    batch_size = 4
    n_points = 3000
    lr = 1e-4
    device = "cuda:"+args.gpu
    print("\ndevice: {}".format(device))

    model = PSGN(encoder = 'conditioning', decoder = 'psgn_2branch', n_points = n_points).to(device)

    print('Args: ', args)


    # ---------------------- prepare data loader ------------------------------- #
    
    _d_train = Sat2LidarDataset(image_dir = os.path.join(args.data_dir, os.path.join('train', 'image_filtered')), pc_num = n_points, ann_dir = os.path.join(args.data_dir, os.path.join('train', 'annotation_fitted_not_scaled_plane_subsampled_dynamic_padded_cleaned')), mode='train')
    _d_val = Sat2LidarDataset(image_dir = os.path.join(args.data_dir, os.path.join('val', 'image_filtered')), pc_num = n_points, ann_dir = os.path.join(args.data_dir, os.path.join('val', 'annotation_fitted_not_scaled_plane_subsampled_dynamic_padded_cleaned')), mode='val')
    
    d_train = torch.utils.data.DataLoader(_d_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    d_val = torch.utils.data.DataLoader(_d_val, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # -------------------------------------------------------------------------- #
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0) 
    milestones = [200, 800, 1300, 2000]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.3)
    
    start_epoch = 0
    prefix, ext = os.path.splitext(args.ckpt_path)
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
            
        A = time.time()

        train_losses = engine.train_one_epoch(model, optimizer, d_train, device, epoch, args)

        A = time.time() - A

        if epoch > args.warmup_epochs:
            lr_scheduler.step()

        print('Train loss: ', train_losses)

        log_loss_info(writer, 'train', train_losses, epoch)
        
        B = time.time()
        eval_losses = engine.evaluate(model, d_val, device, False, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))
        print('Validation loss: ', eval_losses)

        log_loss_info(writer, 'val', eval_losses, epoch)

        engine.save_ckpt(model, optimizer, lr_scheduler, trained_epoch, args.ckpt_path, eval_info=str(eval_losses))

        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 2
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.remove(ckpts[i])
        
    # -------------------------------------------------------------------------- #

    writer.close()

    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="sat2lidar")
    parser.add_argument("--data-dir", default="../dataset/mrcnn_plus_graphx_not_scaled")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results", default="./results/val.res")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.ckpt_path is None:
        args.ckpt_path = "./PSGN_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
    
    main(args)
    
    