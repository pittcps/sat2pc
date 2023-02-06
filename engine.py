import time
import torch
import torch.nn.functional as F
import os
import glob
import re
import graphx
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

    # it will create many checkpoint files during training, so delete some.
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    n = 2
    if len(ckpts) > n:
        for i in range(len(ckpts) - n):
            os.remove(ckpts[i])
            
    if 'best' in kwargs and kwargs['best']==True:
        prefix = prefix+"_best"    
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)

    torch.save(checkpoint, ckpt_path)

    # it will create many checkpoint files during training, so delete some.
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    n = 2
    if len(ckpts) > n:
        for i in range(len(ckpts) - n):
            os.remove(ckpts[i])
    

def train_one_epoch(model, optimizer, data_loader, device, epoch, alpha, args):

    total_emd_loss = 0
    total_graphx_loss = 0
    total = 0

    model.train()

    lr_scheduler = None

    warm_up_epochs = args.warmup_epochs

    if epoch < warm_up_epochs:
        warmup_iters = warm_up_epochs * len(data_loader) - 1
        warmup_factor = 1. / warmup_iters
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


    num_samples = 0
    for i, (init_pc, grey_image, target) in enumerate(data_loader):

        num_samples += 1#len(target)
        init_pc = [torch.tensor(pc).to(device, dtype=torch.float32) for pc in init_pc]
        grey_image = [torch.tensor(im).to(device, dtype=torch.float32) for im in grey_image]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]


        graphx_res, final_out, losses = model(init_pc, grey_image, target)
        total_loss = losses['graphx_loss'] + alpha * losses['refinement_loss']

        total_emd_loss += losses['refinement_loss']
        total_graphx_loss += losses['graphx_loss']
        total += total_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        #if num_iters % args.print_freq == 0:
        #    print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))




    total_emd_loss /= num_samples
    total_graphx_loss /= num_samples
    total /= num_samples
    output = dict(refinement_loss=total_emd_loss, graphx_loss=total_graphx_loss, total_loss=total) 

    return output
            

def evaluate(model, data_loader, device, alpha, is_test, args):
    total_emd_loss = 0
    total_graphx_loss = 0
    total = 0
    num_samples = 0
    worst_graphx_loss = 0
    best_graphx_loss = 999999999
    worst_idx = -1
    best_idx = -1
    pred = []
    gt = []
    for i, (init_pc, grey_image, target) in enumerate(data_loader):
        num_samples += 1#len(target)
        init_pc = [torch.tensor(pc).to(device, dtype=torch.float32) for pc in init_pc]
        grey_image = [torch.tensor(im).to(device, dtype=torch.float32) for im in grey_image]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        with torch.no_grad():
            graphx_res, final_out, losses = model(init_pc, grey_image, target)
            total_loss = losses['graphx_loss'] + alpha * losses['refinement_loss']

            total_emd_loss += losses['refinement_loss']
            total_graphx_loss += losses['graphx_loss']
            if losses['graphx_loss'] < best_graphx_loss:
                best_idx = i
                best_graphx_loss = losses['graphx_loss'] 

            if losses['graphx_loss'] > worst_graphx_loss:
                worst_idx = i
                worst_graphx_loss = losses['graphx_loss'] 
            
            for j in range(len(target)):
                pred.append(final_out[j])
                gt.append(target[j]['lidar'])

            total += total_loss

    total_emd_loss /= num_samples
    total_graphx_loss /= num_samples
    total /= num_samples
    output = dict(refinement_loss=total_emd_loss, graphx_loss=total_graphx_loss, total_loss=total) 

    if is_test == True:
        results, targets = generate_results(model, data_loader, device, args)
        print('Best loss = {}, sample name {}'.format(best_graphx_loss, targets[best_idx]["image_id"].item()))
        print('Worst loss = {}, sample name {}'.format(worst_graphx_loss, targets[worst_idx]["image_id"].item()))





        chamfer_loss  = graphx.get_chamfer(pred, gt)
        emd_loss  = graphx.get_emd(pred, gt)

        output.update(final_chamfer_loss = chamfer_loss)
        output.update(final_emd_loss = emd_loss)



        pred = utility.remove_padding(pred)
        gt = utility.remove_padding(gt)

        no_pad_chamfer_loss  = graphx.get_chamfer(pred, gt)
        no_pad_emd_loss  = graphx.get_emd(pred, gt)

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

    for i, (init_pc, grey_image, targets) in enumerate(data_loader):
        init_pc = [torch.tensor(pc).to(device, dtype=torch.float32) for pc in init_pc]
        grey_image = [torch.tensor(im).to(device, dtype=torch.float32) for im in grey_image]
        #target = {k: v.to(device) for k, v in target.items()}
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        graphx_res, final_out, losses = model(init_pc, grey_image, targets)
        graphx_res = graphx_res.transpose(1, 2)


        predictions = [{targets[i]["image_id"].item(): {'graphx_out': r.cpu() for r in graphx_res}} for i in range(len(targets))]
        for i in range(len(targets)):
            predictions[i][targets[i]["image_id"].item()]['final_out'] = final_out[i]

        results.extend(predictions)
        gt.extend(targets)
     
    #print(results)
    torch.save(results, args.results)
        
    return results, gt
    

