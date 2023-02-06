import os
import torch
from sat2lidar_dataset import Sat2LidarDataset
import engine
from psgn import PSGN
    
def collate_fn(batch):
    return tuple(zip(*batch))

def main(args):
    batch_size = 1
    n_points = 3000
    device = "cuda:"+args.gpu
    print("\ndevice: {}".format(device))


    model = PSGN(encoder = 'conditioning', decoder = 'psgn_2branch', n_points = n_points).to(device)
        
    # ---------------------- prepare data loader ------------------------------- #
    _d_test = Sat2LidarDataset(image_dir = os.path.join(args.data_dir, os.path.join(args.dataset_split, 'image_filtered')), pc_num = n_points, ann_dir = os.path.join(args.data_dir, os.path.join(args.dataset_split, 'annotation')), mode=args.dataset_split, augment=False)
    d_test = torch.utils.data.DataLoader(_d_test, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    
    # -------------------------------------------------------------------------- #
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

    losses = engine.evaluate(model, d_test, device, True, args)

    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="sat2lidar")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--data-dir", default="./datasets/")
    
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results", default="./results/test.res")
    args = parser.parse_args()

    if args.dataset_split != 'test':
        args.results="./results/" + args.dataset_split + ".res"
    
    if args.ckpt_path is None:
        args.ckpt_path = "./graphx_{}.pth".format(args.dataset)
    
    main(args)
    
    