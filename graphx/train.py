import argparse

parser = argparse.ArgumentParser('GraphX-convolution')
parser.add_argument('config_file', type=str, help='config file to dictate training/testing')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import neuralnet_pytorch.gin_nnt as gin

from networks import *
from data_loader_train import ShapeNet, collate

config_file = args.config_file
gin.external_configurable(CNN18Encoder, 'cnn18_enc')
gin.external_configurable(PointCloudEncoder, 'pc_enc')
gin.external_configurable(PointCloudDecoder, 'pc_dec')
gin.external_configurable(PointCloudResDecoder, 'pc_resdec')
gin.external_configurable(PointCloudResGraphXDecoder, 'pc_resgraphxdec')
gin.external_configurable(PointCloudResGraphXUpDecoder, 'pc_upresgraphxdec')
gin.external_configurable(PointCloudResLowRankGraphXUpDecoder, 'pc_upreslowrankgraphxdec')


@gin.configurable('GraphX')
def train_valid(data_root, name, img_enc, pc_enc, pc_dec, optimizer, scheduler, adain=True, projection=True,
                decimation=None, color_img=False, n_points=250, bs=4, lr=5e-5, weight_decay=1e-5, gamma=.3,
                milestones=(5, 8), n_epochs=10, print_freq=1000, val_freq=10000, checkpoint_folder=None):
    if decimation is not None:
        pc_dec = partial(pc_dec, decimation=decimation)
    print("Number of points = ", n_points)

    net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 224, 224), (bs, n_points, 3), img_enc, pc_enc, pc_dec, adain=adain, projection=projection, weight_decay=None)

    print(net)
    solver = T.optim.Adam(net.trainable, 1e-4, weight_decay=0) if optimizer is None \
        else optimizer(net.trainable, lr, weight_decay=weight_decay)
    scheduler = scheduler(solver, milestones=milestones, gamma=gamma) if scheduler is not None else None

    train_data = ShapeNet(path=data_root, grayscale=not color_img, type='train', n_points=n_points)

    sampler = WeightedRandomSampler(train_data.sample_weights, len(train_data), True)

    train_loader = DataLoader(train_data, batch_size=bs, num_workers=1, collate_fn=collate, drop_last=True,
                              sampler=sampler)

    val_data = ShapeNet(path=data_root, grayscale=not color_img, type='val', n_points=n_points)

    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=1, collate_fn=collate, drop_last=True)
    
    mon.model_name = name
    mon.print_freq = print_freq
    mon.num_iters = len(train_data) // bs
    mon.set_path(checkpoint_folder)
    mon.best_val_loss = float('inf')
    if checkpoint_folder is None:
        backups = os.listdir('.')
        mon.backup(backups, ignore=('results', '*.pyc', '__pycache__', '.idea'))
        mon.dump_rep('network', net)
        mon.dump_rep('optimizer', solver)
        if scheduler is not None:
            mon.dump_rep('scheduler', scheduler)

        def model_evaluator():

            def save_checkpoint():
                states = {
                    'states': mon.epoch,
                    'model_state_dict': net.state_dict(),
                    'opt_state_dict': solver.state_dict()
                }
                if scheduler is not None:
                    states['scheduler_state_dict'] = scheduler.state_dict()

                mon.dump(name='training.pt', obj=states, method='torch', keep=2)
            
            # eval model
            net.eval()
            total_loss = 0
            cnt = 0
            with T.set_grad_enabled(False):
                for itt, batch in enumerate(val_loader):

                    batch = utils.batch_to_device(batch, device=args.gpu)
                    loss = net.eval_procedure(batch, reduce = 'mean', normalized = False)
                    total_loss += loss
                    cnt += 1
                    
            loss = total_loss/cnt

            if loss < mon.best_val_loss:
                mon.best_val_loss = loss
            
            save_checkpoint()

        mon.schedule(model_evaluator, when=mon._end_epoch_)
        print('Training...')
    else:
        print("Loading weights...")
        states = mon.load('training.pt', method='torch', map_location='cpu')
        net.load_state_dict(states['model_state_dict'])
        net = net.to(args.gpu)
        net.eval()
        total_loss = 0
        cnt = 0
        with T.set_grad_enabled(False):
            for itt, batch in enumerate(val_loader):

                batch = utils.batch_to_device(batch, device=args.gpu)
                loss = net.eval_procedure(batch, reduce = 'mean', normalized = False)
                total_loss += loss
                cnt += 1
                
        mon.best_val_loss = total_loss/cnt

        def model_evaluator():

            def save_checkpoint():
                states = {
                    'states': mon.epoch,
                    'model_state_dict': net.state_dict(),
                    'opt_state_dict': solver.state_dict()
                }
                if scheduler is not None:
                    states['scheduler_state_dict'] = scheduler.state_dict()

                mon.dump(name='training.pt', obj=states, method='torch', keep=2)
            
            # eval model
            net.eval()
            total_loss = 0
            cnt = 0
            with T.set_grad_enabled(False):
                for itt, batch in enumerate(val_loader):

                    batch = utils.batch_to_device(batch, device=args.gpu)
                    loss = net.eval_procedure(batch, reduce = 'mean', normalized = False)
                    total_loss += loss
                    cnt += 1
                    
            loss = total_loss/cnt

            if loss < mon.best_val_loss:
                mon.best_val_loss = loss
            
            save_checkpoint()

        mon.schedule(model_evaluator, when=mon._end_epoch_)


        print('Resume from epoch %d...' % mon.epoch)
    print('CUDA DEVICE: ', torch.cuda.current_device())

    mon.run_training(net, solver, train_loader, n_epochs, scheduler=scheduler, eval_loader=val_loader,
                     valid_freq=val_freq, reduce='mean', device = args.gpu)
    print('Training finished!')


if __name__ == '__main__':
    gin.parse_config_file(config_file)
    train_valid()
