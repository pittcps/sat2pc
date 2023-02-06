import torch
from torch import nn
import torch.nn.functional as F
import utility

class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x

class Sat2LidarNetwork(nn.Module):
    def __init__(self, graphx_model, pc_num, device):
        super().__init__()

        self.pc_num = pc_num
        self.graphx = graphx_model
        self.res = PointNetRes()
        self.decive = device

    def forward(self, init_pc, grey_images, targets=None):

        graphx_res = []

        lidar_targets = [t['lidar'] for t in targets]

        
        grey_images = torch.stack(grey_images, dim=0)

        init_pc = torch.stack(init_pc, dim=0)
        batch = [init_pc, grey_images, lidar_targets]
        graphx_loss, graphx_res = self.graphx.train_procedure(batch)
        graphx_res = graphx_res.transpose(2, 1)

        delta = self.res(graphx_res)
        final_out = graphx_res + delta
        final_out = final_out.transpose(1, 2)

        refinement_loss = utility.calc_emd(final_out, lidar_targets)

        losses = {'graphx_loss': graphx_loss}
        losses.update(refinement_loss=refinement_loss)

        return graphx_res, final_out, losses
