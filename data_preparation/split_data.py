import os
from shutil import copyfile
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='..\datasets\\extended_dataset\\aggregated\\whole')
parser.add_argument('--output', type=str, default='..\datasets\\extended_dataset\\aggregated\\splitted')
args = parser.parse_args()

splits = ["train", "test", "val"]
percentage = [0.7, 0.15, 0.15]

os.makedirs(args.output, exist_ok=True)
for m in splits:
    dir = os.path.join(args.output, m)
    os.makedirs(dir, exist_ok=True)
    img_dir = os.path.join(dir, "image_filtered") 
    os.makedirs(img_dir, exist_ok=True)
    ann_dir = os.path.join(dir, "annotation") 
    os.makedirs(ann_dir, exist_ok=True)

files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.input, 'image_filtered')) if f.endswith('.png')]
random.shuffle(files)

total_samples = len(files)
split_end_sample = [int(total_samples*percentage[0]), int(total_samples*(percentage[0] + percentage[1])) + 1, total_samples]
print(split_end_sample)
for i in range(0, split_end_sample[0]):
    f = files[i]
    src = os.path.join(os.path.join(args.input, 'image_filtered'), f + '.png')
    dst = os.path.join(os.path.join(args.output, "train\image_filtered"), f + '.png')
    copyfile(src, dst)

    src = os.path.join(os.path.join(args.input, 'annotation_padded'), f + '.json')
    dst = os.path.join(os.path.join(args.output, "train\\annotation"), f + '.json')
    copyfile(src, dst)
        
for i in range(split_end_sample[0], split_end_sample[1]):
    f = files[i]
    src = os.path.join(os.path.join(args.input, 'image_filtered'), f + '.png')
    dst = os.path.join(os.path.join(args.output, "test\\image_filtered"), f + '.png')
    copyfile(src, dst)

    src = os.path.join(os.path.join(args.input, 'annotation_padded'), f + '.json')
    dst = os.path.join(os.path.join(args.output, "test\\annotation"), f + '.json')
    copyfile(src, dst)


for i in range(split_end_sample[1], split_end_sample[2]):
    f = files[i]
    src = os.path.join(os.path.join(args.input, 'image_filtered'), f + '.png')
    dst = os.path.join(os.path.join(args.output, "val\\image_filtered"), f + '.png')
    copyfile(src, dst)

    src = os.path.join(os.path.join(args.input, 'annotation_padded'), f + '.json')
    dst = os.path.join(os.path.join(args.output, "val\\annotation"), f + '.json')
    copyfile(src, dst)