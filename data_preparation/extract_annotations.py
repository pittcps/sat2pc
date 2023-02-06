import os
import json
from PIL import Image, ImageDraw
from shapely import geometry
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='.\dataset\\raw\FL_2')
parser.add_argument('--output', type=str, default='.\dataset\\procces\FL_2')
args = parser.parse_args()

def load_json(input):
    file_addr = os.path.join(input, 'via_region_data.json')
    f = open(file_addr)
    ann = json.load(f)
    f.close()
    return ann

def get_bbox(xy):
    maxi = list(map(max, zip(*xy)))
    mini = list(map(min, zip(*xy)))
    return {'x':mini[0], 'y':mini[1], 'w':maxi[0]-mini[0], 'h':maxi[1] - mini[1]}

def get_mask(xy, im_size):
    msk = [[None]*im_size[1] for _ in range(im_size[0])]
    points = [geometry.Point(x,y) for x, y in xy]
    points.append(points[0])
    poly = geometry.Polygon([[p.x, p.y] for p in points])
    cnt = 0
    for i in range(0, im_size[0]):
        for j in range(0, im_size[1]):
            p =  geometry.Point(i,j)
            if poly.covers(p):
                cnt += 1
                msk[i][j] = 1
            else:
                msk[i][j] = 0
    return msk, poly    

def draw_msk(img, msk, poly):
    show_poly = False
    show_mask = True
    if show_poly:    
        img1 = ImageDraw.Draw(img)  
        x, y = poly.exterior.coords.xy
        poly = [(a, b) for a, b in zip(x, y)]
        img1.polygon(poly, fill =(255, 0, 0), outline ="blue") 
        img.show()
    
    if show_mask:
        pixels = img.load() # create the pixel map
        msk = np.array(msk)
        x = []
        y = []

        for i in range(0, img.size[0]): 
            for j in range(0, img.size[1]): 
                if msk[i][j] == 1:
                    x.append(i)
                    y.append(j)
        
        xy = zip(x, y)

        for x, y in xy:
            pixels[x,y] = (255, 0, 0) # set the colour accordingly
        img.show()
        

def process_image(file_res_dir, name, img, regions, input):
    if len(regions) == 0:
        return
    im_size = img.size
    dic = {}
    dic['filename'] = name
    dic['location'] = input 
    dic['labels'] = []
    dic['bboxes'] = []
    dic['masks'] = []
    dic['points'] = []
    for r in regions:
        if regions[r]['region_attributes']['Structure'] == 'roof':
            if  regions[r]['shape_attributes']['name'] == 'polygon':
                x = regions[r]['shape_attributes']['all_points_x']
                y = regions[r]['shape_attributes']['all_points_y']
            elif regions[r]['shape_attributes']['name'] == 'rect':
                x0 = regions[r]['shape_attributes']['x']
                y0 = regions[r]['shape_attributes']['y']
                w = regions[r]['shape_attributes']['width'] 
                h = regions[r]['shape_attributes']['height']
                x = [x0, x0 + w, x0 + w, x0]
                y = [y0, y0, y0 + h, y0 + h]
            else:
                assert(False) #invalid shape
            
            label = regions[r]['region_attributes']['Type']
            all_l = ['', 'N', 'E', 'W', 'S', 'NW', 'NE', 'SW', 'SE', 'flat', 'NNW', 'WNW', 'WSW', 'SSW', 'SSE', "ESE", "ENE", "NNE"]
            ind = all_l.index(label)

            dic['labels'].append(label)

            xy = list(zip(x, y))
            dic['points'].append(xy)

            bbox = get_bbox(xy)
            dic['bboxes'].append(bbox)

            msk, poly = get_mask(xy, im_size)
            dic['masks'].append(msk)

            #draw_msk(img, msk, poly)

    ann_dir = os.path.join(os.path.join(file_res_dir, 'annotation'), name + '.json')
    with open(ann_dir, "w") as outfile: 
        json.dump(dic, outfile)
    img.save(os.path.join(os.path.join(os.path.join(file_res_dir, 'image'), name + '.png')))

def extract_annotations(input, output):
    print("Extracting annotations")
    ann = load_json(input)
    os.makedirs(os.path.join(output, 'image'), exist_ok=True)
    os.makedirs(os.path.join(output, 'annotation'), exist_ok=True)


    files_ = os.listdir(os.path.join(input, 'image'))
    for d in ann:
        name = ann[d]['filename'] 
        if name not in files_:
            continue
        print(name)
        img = Image.open(os.path.join(input, os.path.join('image', name)))
        process_image(output, os.path.splitext(name)[0], img, ann[d]['regions'], input)