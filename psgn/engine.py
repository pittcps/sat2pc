import os
import time
import torch
import torch.nn.functional as F
import utility

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def save_ckpt(model, optimizer, lr_scheduler, epochs, ckpt_path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(ckpt_path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, ckpt_path)
    

def train_one_epoch(model, optimizer, data_loader, device, epoch, args):

    total = 0

    model.train()

    lr_scheduler = None

    warm_up_epochs = args.warmup_epochs

    if epoch < warm_up_epochs:
        warmup_iters = warm_up_epochs * len(data_loader) - 1
        warmup_factor = 1. / warmup_iters
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


    num_samples = 0
    for i, (r_vec, image, target) in enumerate(data_loader):
        num_samples += 1#len(target)
        r_vec = torch.tensor(r_vec).to(device, dtype=torch.float32)
        image = torch.tensor(image).to(device, dtype=torch.float32)
        gt = [t['lidar'].to(device)  for t in target]

        pred_points = model(r_vec, image)

        c_loss = utility.calc_chamfer(pred_points, gt)

        total += c_loss

        c_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()


    total /= num_samples
    output = dict(total_loss=total) 

    return output
            

def evaluate(model, data_loader, device, is_test, args):
    total = 0
    num_samples = 0
    worst_loss = 0
    best_loss = 999999999
    worst_idx = -1
    best_idx = -1
    all_gt = []
    all_pred = []
    for i, (r_vec, image, target) in enumerate(data_loader):
        num_samples += 1#len(target)
        r_vec = torch.tensor(r_vec).to(device, dtype=torch.float32)
        image = torch.tensor(image).to(device, dtype=torch.float32)
        gt = [t['lidar'].to(device)  for t in target]

        with torch.no_grad():
            pred_points = model(r_vec, image)

            c_loss = utility.calc_chamfer(pred_points, gt)

            if c_loss < best_loss:
                best_idx = i
                best_loss = c_loss

            if c_loss > worst_loss:
                worst_idx = i
                worst_loss = c_loss 

            total += c_loss

            for j in range(len(target)):
                all_pred.append(pred_points[j])
                all_gt.append(gt[j])

    total /= num_samples

    output = dict(total_loss=total) 

    if is_test == True:
        results, targets = generate_results(model, data_loader, device, args)
        print('Best loss = {}, sample name {}'.format(best_loss, targets[best_idx]["image_id"].item()))
        print('Worst loss = {}, sample name {}'.format(worst_loss, targets[worst_idx]["image_id"].item()))

        
        chamfer_loss  = utility.calc_chamfer(all_pred, all_gt)
        emd_loss  = utility.calc_emd(all_pred, all_gt)

        output.update(final_chamfer_loss = chamfer_loss)
        output.update(final_emd_loss = emd_loss)



        pred = utility.remove_padding(all_pred)
        gt = utility.remove_padding(all_gt)

        no_pad_chamfer_loss  = utility.calc_chamfer(pred, gt)
        no_pad_emd_loss  = utility.calc_emd(pred, gt)

        outline_iou, _ = utility.get_outline_iou(pred, gt)

        output.update(final_no_pad_chamfer_loss = no_pad_chamfer_loss)
        output.update(final_no_pad_emd_loss = no_pad_emd_loss)
        output.update(outline_iou = outline_iou)

    

    return output
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
        
    results = []
    gt = []
    model.eval()

    for i, (r_vec, image, target) in enumerate(data_loader):
        r_vec = torch.tensor(r_vec).to(device, dtype=torch.float32)
        image = torch.tensor(image).to(device, dtype=torch.float32)

        torch.cuda.synchronize()
        pred_points = model(r_vec, image)
        # graphx_res = graphx_res.transpose(1, 2)
        predictions = [{target[i]["image_id"].item(): {'final_out': r.cpu() for r in pred_points}} for i in range(len(target))]

        results.extend(predictions)
        gt.extend(target)
     
    torch.save(results, args.results)
        
    return results, gt
    

