import scipy.io as sio
import os
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import glob


EXPR_PATH = ['/Volumes/LTQ/QYM data/data/gal nacl02_gal nacl03_gal nacl04_gly nacl02 20160711',
             '/Volumes/LTQ/QYM data/data/raf caf4_raf dtt05_gal dtt05_gal nacl05 20160703'
             ]


def correct_image_path(path_imageBF):
    pl = path_imageBF.split(':')
    correct_path = '/Volumes/LTQ' + pl[1]
    correct_path = reduce(lambda a,b: a+'/'+b,correct_path.split('\\'))
    return correct_path


def load_csg(filename_csg):
    print('loading'+filename_csg)
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


def extract_xypoint(filepath_csg):
    # extract data from a single xypoint's images
    seg_data = load_csg(filepath_csg)
    n_frame = seg_data['cellsegperim'].shape[1]
    seg_ims = list()
    for frame in range(0, n_frame, int(n_frame/5)):     # for each xypoint, extract 5 images
        seg_im = get_seg_im(seg_data, frame)
        seg_ims.append(seg_im)
    return seg_ims

def extract_expr(expr_path):
    csg_paths = glob.glob(expr_path + '/*.csg')
    seg_im_list = list()
    for csg_file in csg_paths:
        seg_ims = extract_xypoint(csg_file)
        seg_im_list.extend(seg_ims)
    return seg_im_list


def get_seg_im_data():
    seg_im_data = list()
    for path in EXPR_PATH:
        seg_im_data.extend(extract_expr(path))
    return seg_im_data
