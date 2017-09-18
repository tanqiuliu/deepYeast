import scipy.io as sio
import os
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
import skimage
from skimage import draw
#import cv2

IMAGE_SHAPE = (512, 512, 1)
WINDOW_SHAPE = (50, 50)
STEP_SIZE = (7, 7)

def correct_image_path(path_imageBF):
    pl = path_imageBF.split(':')
    correct_path = 'I:' + pl[1]
    return correct_path


def load_csg(filename_csg):
    mat_contents = sio.loadmat(filename_csg)
    hh = mat_contents['hh']
    val = hh[0, 0]
    seg_data = dict()
    seg_data['cellsegperim'] = val['cellsegperim']
    seg_data['filenameBF'] = val['filenameBF']
    seg_data['path_imageBF'] = str(val['pathnameBF'][0])
    # 下步仅用于矫正不同机器上驱动器名称的误差
    seg_data['path_imageBF'] = correct_image_path(seg_data['path_imageBF'])
    return seg_data


def transform_cellseg(cellseg):
    cellsegs = list()
    for i in range(cellseg.shape[0]):
        seg = cellseg[i, 0]
        if(seg.shape[1]==2):
            cellsegs.append(seg)
    return cellsegs


def get_seg_im(seg_data, idx):
    seg_im = dict()
    seg_im['cellseg'] = transform_cellseg(seg_data['cellsegperim'][0, idx])
    seg_im['filenameBF'] = str(seg_data['filenameBF'][0, idx][0])
    image_file = os.path.join(seg_data['path_imageBF'], seg_im['filenameBF'])
    seg_im['imageBF'] = np.array(Image.open(image_file))
    return seg_im


def segperim_generator(seg_data):
    for i in range(seg_data['cellsegperim'].shape[1]):
        seg_im = get_seg_im(seg_data, i)
        yield seg_im


def plot_cellseg(seg_im):
    colormap = mpl.cm.gray
    plt.imshow(seg_im['imageBF'], cmap=colormap)
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        plt.plot(seg[:, 1], seg[:, 0], 'r')
        plt.plot(find_center(seg)[1], find_center(seg)[0], 'r*')
    plt.show()


def find_center(seg):
    c_mean = np.mean(seg[:, 1])
    r_mean = np.mean(seg[:, 0])
    return (int(r_mean), int(c_mean))





"""
def get_circle(seg):
    circle = np.zeros(IMAGE_SHAPE)
    circle[seg[:, 0], seg[:, 1]] = 1
    return circle


def get_circles(seg_im):
    circles = np.zeros(IMAGE_SHAPE)
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx]
        circles[seg[:, 0], seg[:, 1]] = 1
    return circles
"""

"""
seg_data = load_csg('E:/LTQ work/tanglab/deepYeast/xy01 1-120.csg')
seg_im = next(segperim_generator(seg_data))
plot_cellseg(seg_im)
"""

"""
# 看来还是不能保存为.mat， 不然再次打开内部结构和数据类型就乱了
def load_seg_im_data(filename):
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
"""

def load_data(filename_list):
    Xy_list = list()
    for filename in filename_list:
        Xy_list.append(np.loadtxt(filename, dtype=np.int32, comments='#', delimiter=' '))
    Xy = np.vstack(tuple(Xy_list))
    np.random.shuffle(Xy)
    X = Xy[:, 1:]
    y = Xy[:, 0]
    X = X.reshape(X.shape[0], WINDOW_SHAPE[0], WINDOW_SHAPE[1])
    return X, y


def load_rect_data(filename_list):
    Xy_list = list()
    for filename in filename_list:
        Xy_list.append(np.loadtxt(filename, dtype=np.int32, comments='#', delimiter=' '))
    Xy = np.vstack(tuple(Xy_list))
    np.random.shuffle(Xy)
    X = Xy[:, 4:]
    y = Xy[:, 0:4]
    X = X.reshape(X.shape[0], WINDOW_SHAPE[0], WINDOW_SHAPE[1])
    return X, y





def save_image(data, path):
    for idx in range(data.shape[0]):
        im = data[idx]
        img = Image.fromarray(np.uint16(im))
        img.save(os.path.join(path, '%s.tif'%idx))


def plot_rect(imageBF, vertex):
    colormap = mpl.cm.gray
    plt.imshow(imageBF, cmap=colormap)
    (r1, r2, c1, c2) = vertex
    plt.plot(np.ones(r2-r1)*c1, np.array(range(r1, r2)), 'r')
    plt.plot(np.ones(r2-r1)*c2, np.array(range(r1, r2)), 'r')
    plt.plot(np.array(range(c1, c2)), np.ones(c2-c1)*r1, 'r')
    plt.plot(np.array(range(c1, c2)), np.ones(c2-c1)*r2, 'r')
    plt.xlim(0, WINDOW_SHAPE[1])
    plt.ylim(WINDOW_SHAPE[0], 0)
    plt.show()

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
