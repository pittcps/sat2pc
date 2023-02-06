import json
import laspy
import math
import shapely
from PIL import Image
from laspy.file import File as LasFile
import numpy as np
from shapely.geometry import Polygon, Point
from pyproj import Proj, transform 
import pptk
import MercatorProjection
from matplotlib import cm
import os
import alphashape
import scipy.spatial as spatial
from statistics import mean, median, stdev
import matplotlib.pyplot as plt
import matplotlib


# Uses an image to extract appropriate RGB color for each point in the point cloud. This is done based on the point's location and the closest pixel of the image to that location.
def plot_3d_pc_with_imge_overlay(pc, img, center_point, zoom, w, h, srs, offset, scale, adjustment_offset):
    rgb = []

    for p in pc:
        x, y, z = p
        point_latlon = MercatorProjection.G_Point(x, y)
        px, py = MercatorProjection.get_pixel_xy_from_lat_lon(center_point, point_latlon, zoom, w, h, srs, offset, scale, adjustment_offset)
        pixel_color = [float(v/255.) for v in img.getpixel((int(px), int(py)))]
        rgb.append(pixel_color)

    plot_3d(pc, rgb=rgb)

def plot_density_distribution(pred, gt, r, title):
    matplotlib.use('TkAgg')
    tree = spatial.KDTree(np.array(pred))
    gt_tree = spatial.KDTree(np.array(gt))
    neighbors = tree.query_ball_tree(gt_tree, r)
    freq = np.array(list(map(len, neighbors)))
    D = freq / (4/3*math.pi*pow(r,3))

    plt.hist(D, color = 'blue', edgecolor = 'black', bins = 10)
    plt.title(title)
    plt.xlabel('Density')
    plt.ylabel('Number of Points')
    plt.show()

def plot_two_density_distribution(pred1, pred2, gt, r, title, name1, name2):
    print("len gt ", gt.shape)
    matplotlib.use('TkAgg')
    tree = spatial.KDTree(np.array(pred1))
    tree2 = spatial.KDTree(np.array(pred2))
    gt_tree = spatial.KDTree(np.array(gt))
    neighbors = gt_tree.query_ball_tree(tree, r)

    freq = np.array(list(map(len, neighbors)))
    D1 = freq / (4/3*math.pi*pow(r,3))


    neighbors = gt_tree.query_ball_tree(tree2, r)
    freq = np.array(list(map(len, neighbors)))
    D2 = freq / (4/3*math.pi*pow(r,3))

    neighbors = gt_tree.query_ball_tree(gt_tree, r)
    freq = np.array(list(map(len, neighbors)))
    D3 = freq / (4/3*math.pi*pow(r,3))

    plt.hist([D1, D2, D3], edgecolor = 'black', bins = 10, label=[name1, name2, 'Ground truth'])
    plt.title(title)
    plt.xlabel('Density')
    plt.ylabel('Number of Points')
    plt.legend(loc='upper left')

    plt.show()

def plot_box_chart_density_dist(pred1, pred2, gt, r, title, name1, name2):
    matplotlib.use('TkAgg')
    plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
    plt.rcParams.update({'font.size': 40})
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
    plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
    tree = spatial.KDTree(np.array(pred1))
    tree2 = spatial.KDTree(np.array(pred2))
    gt_tree = spatial.KDTree(np.array(gt))
    neighbors = gt_tree.query_ball_tree(tree, r)
    freq = np.array(list(map(len, neighbors)))
    D1 = freq / (4/3*math.pi*pow(r,3))

    neighbors = gt_tree.query_ball_tree(tree2, r)
    freq = np.array(list(map(len, neighbors)))
    D2 = freq / (4/3*math.pi*pow(r,3))

    neighbors = gt_tree.query_ball_tree(gt_tree, r)
    freq = np.array(list(map(len, neighbors)))
    D3 = freq / (4/3*math.pi*pow(r,3))

    my_dict = {name1: D1, name2: D2, 'Ground truth': D3}

    fig, ax = plt.subplots()
    ax.boxplot(my_dict.values(), 0, '', whis = 100)
    ax.set_xticklabels(my_dict.keys())
    ax.set_ylabel('Density')
    plt.show()

# calculates the density around each point from the ground truth in two predictions and the ground truth itself 
def get_density(pred1, pred2, gt, r):
    tree = spatial.KDTree(np.array(pred1))
    tree2 = spatial.KDTree(np.array(pred2))
    gt_tree = spatial.KDTree(np.array(gt))
    neighbors = gt_tree.query_ball_tree(tree, r)
    freq = np.array(list(map(len, neighbors)))
    D1 = freq / (4/3*math.pi*pow(r,3))

    neighbors = gt_tree.query_ball_tree(tree2, r)
    freq = np.array(list(map(len, neighbors)))
    D2 = freq / (4/3*math.pi*pow(r,3))

    neighbors = gt_tree.query_ball_tree(gt_tree, r)
    freq = np.array(list(map(len, neighbors)))
    D3 = freq / (4/3*math.pi*pow(r,3))

    return D1, D2, D3
    
def get_average_point_density(points, r):
    tree = spatial.KDTree(np.array(points))
    neighbors = tree.query_ball_tree(tree, r)
    freq = np.array(list(map(len, neighbors)))
    D = freq / (4/3*math.pi*pow(r,3))
    return mean(D), stdev(D)

def draw_masks(masks):
    colors = [(255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 125, 0),
    (0, 255, 125),
    (255, 255, 255)]
    #img = img.convert('RGBA')
    pc = []
    rgb = []
    for i in range(len(masks)):
        mask = masks[i]

        on = (mask > 0.).nonzero()
        x = on[0].tolist()
        y = on[1].tolist()


        for j, _x in enumerate(x):
            _y = y[j]

            pc.append([_x, _y, 0])
            rgb.append(colors[i%9])
                
    pc = np.asarray(pc)
    plot_3d(pc, rgb=rgb)

def get_pointcolud_outline(pc, l=0.1):
    if pc.shape[1] != 3:
        pc = pc.transpose(0, 1)
    pc_2d = pc[:, :2]
    poly = alphashape.alphashape(pc_2d, l)
    return poly

def get_lat_lon_point(lat_center, lon_center, lat, lon, zoom, width, height):
    parallel_multiplier = math.cos(lat_center * math.pi / 180)
    degrees_per_pixel_x = 360 / math.pow(2, zoom + 8 + 1.15)
    degrees_per_pixel_y = 360 / math.pow(2, zoom + 8 + 1.15) * parallel_multiplier

    x = (lon - lon_center) / degrees_per_pixel_x + (width / 2.)
    y = (lat_center - lat) / degrees_per_pixel_y + (height / 2.)

    return (x, y)

def get_SI_square(centerlat, centerlon, srs, zoom, width, height, using_googlemaps):
    centerPoint = MercatorProjection.G_LatLng(centerlat, centerlon)
    z = zoom
    if using_googlemaps == False:
        z = z + 1
    #corners are in lat lon
    corners = MercatorProjection.getCorners(centerPoint, z, width, height) 

    SWLat, SWLon, NELat, NELon = corners['S'], corners['W'], corners['N'], corners['E']  #getCorners(centerlat, centerlon, zoom, width, height)

    tr_x, tr_y = get_xy_from_lat_lon(NELat, NELon, srs)
    bl_x, bl_y = get_xy_from_lat_lon(SWLat, SWLon, srs)

    br_x = tr_x
    br_y = bl_y

    tl_x = bl_x
    tl_y = tr_y

    square = [(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y), (tl_x, tl_y)]
    return square

def load_json(path):
    f = open(path)
    ann = json.load(f)
    f.close()
    return ann

def load_image(path):
    return Image.open(path).convert("RGB")    

def load_lidar(path):
    inFile = LasFile(path, mode='r')
    cords = [inFile.X,inFile.Y,inFile.Z]
    lidar = np.array(cords)
    h = inFile.header
    scale = h.scale
    offset = h.offset

    # TODO: Add these to lidar files
    scale = [0.01, 0.01, 0.01]
    inFile.close()
    return lidar, offset, scale

def get_center_latlon_from_wayid(way):
    poly = way['nodes']
    polygon = Polygon(poly)
    point = polygon.representative_point()
    # centerlat, centerlon
    lat, lon = point.x, point.y
    return poly, lat, lon

def get_lat_lon_from_xy(x, y, srs):
    proj_in = Proj("epsg:{}".format(srs))
    proj_out = Proj('epsg:3857')
    x1, y1 = transform(proj_in, proj_out, x, y)
    lng, lat = proj_out(x1, y1, inverse=True)
    return lat, lng

def get_xy_from_lat_lon(lat, lon, srs):
    proj_in = Proj('epsg:4326')
    proj_out = Proj('epsg:{}'.format(srs))
    x, y = transform(proj_in, proj_out, lat, lon)
    return x, y

def get_pixel_xys(polygons, centerlat, centerlon, zoom, width, height):
    xys = []
    for lat, lon in polygons:
        x, y = get_lat_lon_point(centerlat, centerlon, lat, lon, zoom, width, height)
        xys.append((x, y))
    return xys

def to_lidar_xy(lidar_xy, offset, scale, adjustment_offset):
    raw_xy = [((co_x - offset[0] ) / scale[0] + adjustment_offset[0], (co_y - offset[1]) / scale[1] + adjustment_offset[1]) for (co_x, co_y) in lidar_xy]
    return raw_xy

def show_image(img):
    if isinstance(img, np.ndarray):
        img = img / 2
        img = img + 0.5
        img = img * 255

        img = np.uint8(np.transpose(img, (1, 2, 0)))
        img = img.reshape((224, 224))
        img = Image.fromarray(img)
    img.show()

def save_lidar(lidar, dir, offset, scale):
    header = laspy.header.Header()
    outFile = laspy.file.File(dir, mode='w', header=header)
    outFile.offset = offset
    outFile.scale = scale
    outFile.X = lidar[0]
    outFile.Y = lidar[1]
    outFile.Z = lidar[2]

    outFile.close()

def save_image(img, dir):
    img.save(dir)

def prepare_segments(segments):
    s = []
    for x in segments:
        if x[0] == x[-1]:
            s.append(x)
        else:
            x.append(x[0])
            s.append(x)
    return s

def convert_segments_to_lat_lon(segments, center_point, zoom, w, h, srs, offset, scale, adjustment_offset):
    s = []
    for x in segments:
        points = []
        for p in x:
            lat, lon = MercatorProjection.get_pixel_lat_lon(center_point, p[0], p[1], zoom, w, h)
            px, py = get_xy_from_lat_lon(lat, lon, srs)
            new_p = to_lidar_xy([[px, py]], offset, scale, adjustment_offset)
            points.append(new_p[0])
        s.append(points)
    return s

def project_segments_to_lidar(segments, lidar, wayid, zoom, w, h, srs, offset, scale, adjustment_offset):
    _, center_lat, center_lon = get_center_latlon_from_wayid(wayid)
    center_point = MercatorProjection.G_LatLng(center_lat, center_lon)
    segments = prepare_segments(segments)
    segments = convert_segments_to_lat_lon(segments, center_point, zoom, w, h, srs, offset, scale, adjustment_offset)
    polys = []

    for s in segments:
        polys.append(Polygon(s))

    rgb = []
    num = len(lidar[0])
    c = 0
    for i in range(num):
        lat = lidar[0][i]
        lon = lidar[1][i]

        p = Point(lat, lon)
        found = 0
        for j, poly in enumerate(polys):
            if poly.contains(p):
                c += 1
                if j%3 == 0:
                    rgb.append([float(0) / 255, float(255) / 255, float(0)/255])
                elif j%3 == 1:
                    rgb.append([float(255) / 255, float(0) / 255, float(0)/255])
                else:
                    rgb.append([float(0) / 255, float(0) / 255, float(255)/255])
                found = 1
                break
        if found == 0:
            rgb.append([float(255) / 255, float(255) / 255, float(255)/255])

    plot_3d(lidar, num, rgb)
    input("Press Enter to continue...")

def plot_3d(data, num = None, rgb = None):
    if(len(data) == 3):
        P = np.array(data).T
    else:
        P = data
        
    neighbors = []
    fill_rgb = False
    if rgb is None:
        fill_rgb = True
        rgb = []

    if num is None:
        num = len(P)

    for i in range(0, num):
        if fill_rgb == True:
            rgb.append([0, 1, 0])
        neighbors.append(P[i, :])

    v = pptk.viewer(neighbors, np.array(rgb))
    v.set(point_size=5)

def visualize_3d_pixels(heights, id = None, image = None):
    n_colors = 6
    data = [[], [], []]
    rgb = []
    for i in range(len(heights)):
        for j in range(len(heights[0])):
            ind = i * len(heights[0]) + j

            if image != None:
                if j >= image.size[0] or i >= image.size[1]:
                    pixel_color = [0, 0, 0]
                else:
                    pixel_color = [float(v/255.) for v in image.getpixel((j,i))]
                rgb.append(pixel_color)
                data[0].append(i)
                data[1].append(j)
                data[2].append(heights[i][j])
            elif id == None:
                rgb.append([float(0) / 255, float(255) / 255, float(0)/255])
                data[0].append(i)
                data[1].append(j)
                data[2].append(heights[i][j])
                
            elif id[ind] != -1:
                data[0].append(i)
                data[1].append(j)
                data[2].append(heights[i][j])
                if id[ind]%n_colors == 0:
                    rgb.append([float(0) / 255, float(255) / 255, float(0)/255])
                elif id[ind]%n_colors == 1:
                    rgb.append([float(255) / 255, float(0) / 255, float(0)/255])
                elif id[ind]%n_colors == 2:
                    rgb.append([float(255) / 255, float(0) / 255, float(255)/255])
                elif id[ind]%n_colors == 3:
                    rgb.append([float(0) / 255, float(255) / 255, float(255)/255])
                elif id[ind]%n_colors == 4:
                    rgb.append([float(255) / 255, float(255) / 255, float(0)/255])
                elif id[ind]%n_colors == 5:
                    rgb.append([float(0) / 255, float(0) / 255, float(255)/255])

    num = len(rgb)
    plot_3d(data, num, rgb)

def visualize_3d_array(points, id, lidar):
    data = [[], [], []]
    rgb = []
    num = 0

    for i, x in enumerate(points):

        data[0].append(float(x[0]) *10)  # X
        data[1].append(float(x[1]) *10)  # Y
        data[2].append(float(x[2]) )  # Z

        if id[i]%3 == 0:
            rgb.append([float(0) / 255, float(255) / 255, float(0)/255])
        elif id[i]%3 == 1:
            rgb.append([float(255) / 255, float(0) / 255, float(0)/255])
        else:
            rgb.append([float(0) / 255, float(0) / 255, float(255)/255])

        num = num + 1

    plot_3d(data, num, rgb)

    rgb2 = []
    num2 = 0

    for x in range(len(lidar[0])):

        rgb2.append([float(0) / 255, float(255) / 255, float(0)/255])

        num2 = num2 + 1

    plot_3d(lidar, num2, rgb2)

    input("Press Enter to continue...")

def plot_mask(msk):
    im = Image.fromarray(np.uint8(msk * 255))
    im.show()
