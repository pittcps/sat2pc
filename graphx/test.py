import sys
sys.path.append("../PyTorchEMD/")
import argparse
import pickle5 as pickle
import os
from torch.utils.data import DataLoader
import neuralnet_pytorch.gin_nnt as gin
from .networks import *
from .data_loader_test import ShapeNet, collate
import shutil



gin.external_configurable(CNN18Encoder, 'cnn18_enc')
gin.external_configurable(PointCloudEncoder, 'pc_enc')
gin.external_configurable(PointCloudDecoder, 'pc_dec')
gin.external_configurable(PointCloudResDecoder, 'pc_resdec')
gin.external_configurable(PointCloudResGraphXDecoder, 'pc_resgraphxdec')
gin.external_configurable(PointCloudResGraphXUpDecoder, 'pc_upresgraphxdec')
gin.external_configurable(PointCloudResLowRankGraphXUpDecoder, 'pc_upreslowrankgraphxdec')

@gin.configurable('GraphX')
def test_each_category(data_root, checkpoint_folder, img_enc, pc_enc, pc_dec, color_img=False, n_points=250, **kwargs):
    bs = 1
    print(img_enc)
    mon.print_freq = 1
    mon.set_path(args.ckpt_path)

    states = mon.load("training.pt", method='torch')
    net = PointcloudDeformNet((bs,) + (3 if color_img else 1, 224, 224), (bs, n_points, 3), img_enc, pc_enc, pc_dec)

    if nnt.cuda_available:
        net = net.cuda()

    net.load_state_dict(states['model_state_dict'])
    net.eval()
    result_dir = os.path.join(args.ckpt_path, args.dataset_split)
    os.makedirs(result_dir, exist_ok=True)    

    test_data = ShapeNet(path = data_root, grayscale=not color_img, type=args.dataset_split, n_points=n_points)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=1, collate_fn=collate)

    print('Testing...')
    with T.set_grad_enabled(False):
        for itt, batch in enumerate(test_loader):
            names, init_pc, _image, gt_pc = batch

            if nnt.cuda_available:
                init_pc = init_pc.cuda()
                image = _image.cuda()
                gt_pc = [pc.cuda() for pc in gt_pc] if isinstance(gt_pc, (list, tuple)) else gt_pc.cuda()

            pred_pc = net(image, init_pc)
            pred_pc = pred_pc.cpu().numpy()
            for i in range(len(names)):
                data = [pred_pc[i]]
                with open(os.path.join(result_dir, names[i] + '.out'), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(args.results, exist_ok=True) 
    files = os.listdir(result_dir)
    for f in files:
        shutil.move(os.path.join(result_dir, f), os.path.join(args.results, f))

    print('Testing finished!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GraphX-convolution')
    parser.add_argument('--config', type=str, help='config file to dictate training/testing')
    parser.add_argument("--results", default="./results/test.res")
    parser.add_argument("--ckpt-path")
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
    parser.add_argument('--dataset-split', default='test')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    config_file = args.config
    gin.parse_config_file(config_file)
    test_each_category()
