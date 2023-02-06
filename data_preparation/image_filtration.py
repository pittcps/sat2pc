import sys
import os
from shapely.geometry import Polygon, Point
ROOT_DIR = os.path.abspath("../")
sys.path.append('../')
import data_util

def filter_image(img, segments, plot):
    segments = data_util.prepare_segments(segments)
    polys = []
    for s in segments:
        polys.append(Polygon(s))

    w, h = img.size

    new_img = img.copy()
    pixels = new_img.load()

    for x in range(w): 
        for y in range(h):

            p = Point(x, y)
            found = False
            for j, poly in enumerate(polys):
                if poly.contains(p):
                    found = True
                    break
            if found == False:
                pixels[x, y] = (0, 0, 0)

    if plot == True:
        data_util.show_image(new_img)

    return new_img

def filter_all_images(input, output, use_wayid, plot, srs, zoom, width, height):
    print("Filtering images\n")
    img_addr = os.path.join(output, 'image')
    ann_addr = os.path.join(output, 'annotation')
    lidar_addr = os.path.join(input, 'lidar')
    wayid_addr = os.path.join(input, 'wayids')
    os.makedirs(os.path.join(output, 'image_filtered'), exist_ok=True)

    files = [f for f in os.listdir(img_addr) if f.endswith('.png')]

    for f in files:
        file_basename = os.path.splitext(f)[0]
        print("sample: ", f, "\n")
        img_out_addr = os.path.join(os.path.join(output, 'image_filtered'), file_basename + '.png')
        img = data_util.load_image(os.path.join(img_addr, file_basename + '.png'))
        wayid = data_util.load_json(os.path.join(wayid_addr,  file_basename + '.json'))

        poly, center_lat, center_lon = data_util.get_center_latlon_from_wayid(wayid)

        if use_wayid == False:
            ann = data_util.load_json(os.path.join(ann_addr, file_basename + '.json'))
            segments = ann['points']
        else:
            poly = data_util.get_pixel_xys(poly, center_lat, center_lon, zoom, width, height)
            segments = [poly]

        new_img = filter_image(img, segments, plot)

        if plot == True:
            data_util.show_image(img)

        data_util.save_image(new_img, img_out_addr)