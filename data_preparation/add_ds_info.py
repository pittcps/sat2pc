import json
import os

def add_ds_info(input, ds_id, lidar_srs, img_zoom, img_width, img_height):
    print("adding dataset info to samples: \n")
    ann_dir = os.path.join(input, 'annotation')

    files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]

    for f in files:
        print("sample ", f, "\n")
        ann_path = os.path.join(ann_dir, f)
        fi = open(ann_path)
        ann = json.load(fi)
        fi.close()
        ann['dataset_id'] = ds_id
        ann['lidar_srs'] = lidar_srs
        ann['img_zoom'] = img_zoom
        ann['img_width'] = img_width
        ann['img_height'] = img_height
        with open(ann_path, 'w') as outfile:
            json.dump(ann, outfile)
