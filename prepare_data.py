from util import *
import glob
from skimage.morphology import binary_dilation

EXPR_PATH = ['I:\QYM data\data\gal nacl02_gal nacl03_gal nacl04_gly nacl02 20160711',
             'I:\QYM data\data/raf caf4_raf dtt05_gal dtt05_gal nacl05 20160703',
             'I:\QYM data\data\Gly H2O2009_DTT02_DTT05_Gal004 dtt075 20160627',
             'I:\QYM data\data\SD_NaCl1_Eth_Glu0005 20160612',
             'I:\QYM data\data/002_004Gal 20160513',
             'I:\QYM data\data\SD-DTT075_H2O203_Glu005 20160511',
             'I:\QYM data\data/ura1-his2_3_5_6_8 20160505.nd2'
             ]


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


def load_seg_im_data():
    seg_im_data = list()
    for path in EXPR_PATH:
        seg_im_data.extend(extract_expr(path))
    return seg_im_data


def load_seg_im_data(filename):
    mat_contents = sio.loadmat(filename)
    data = mat_contents['data']
    seg_im_data = list()
    for idx in range(data.shape[1]):
        data_im = data[0, idx][0, 0]
        seg_im = dict()
        seg_im['cellseg'] = data_im['cellseg'].transpose()
        seg_im['filenameBF'] = str(data_im['filenameBF'][0])
        seg_im['imageBF'] = data_im['imageBF']
        seg_im_data.append(seg_im)
    return seg_im_data

# ----------------------------------------------------

def reduce_image_size():
    pass


def cut_window(image, center):
    r1 = int(center[0] - WINDOW_SHAPE[0] / 2)
    r2 = int(center[0] + WINDOW_SHAPE[0] / 2)
    c1 = int(center[1] - WINDOW_SHAPE[1] / 2)
    c2 = int(center[1] + WINDOW_SHAPE[1] / 2)
    return image[r1:r2, c1:c2]


def gen_pos(seg_im):
    r_range = (int(WINDOW_SHAPE[0] / 2) + 1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0] / 2) - 1)
    c_range = (int(WINDOW_SHAPE[1] / 2) + 1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1] / 2) - 1)
    pos_data = list()
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        center = find_center(seg)
        if(center[0] > r_range[0] and center[0] < r_range[1] and center[1] > c_range[0] and center[1] < c_range[1]):
            window = cut_window(seg_im['imageBF'], center)
            pos_data.append(window.reshape(1, WINDOW_SHAPE[0] * WINDOW_SHAPE[1]))
    feat = np.vstack(tuple(pos_data))
    target = np.ones((feat.shape[0], 1))
    return np.hstack((target, feat))


def gen_pos_example(seg_im_data):
    tables = list()
    for seg_im in seg_im_data:
        pos_table = gen_pos(seg_im)
        tables.append(pos_table)
    return np.vstack(tuple(tables)).astype(np.int32)


"""
seg_im_data = load_seg_im_data('seg_im_data.mat')
pos_data_table = gen_pos_example(seg_im_data)
np.savetxt('positive_data.csv', pos_data_table, fmt='%d', header = 'positive_examples, first col is target')
"""

def get_cor_range(seg):
    r1 = np.min(seg[:, 0])
    r2 = np.max(seg[:, 0])
    c1 = np.min(seg[:, 1])
    c2 = np.max(seg[:, 1])
    return (r1, r2, c1, c2)

def gen_mask(seg_im):
    mask =  np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[0]))
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        (r1, r2, c1, c2) = get_cor_range(seg)
        mask[r1:r2, c1:c2] = 1
    return mask.astype(np.int32)


def get_pos_mask(seg_im):
    pos_mask = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[0]))
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        (r, c) = find_center(seg)
        pos_mask[r-int(STEP_SIZE[0]/2):r+int(STEP_SIZE[0]/2), c-int(STEP_SIZE[1]/2):c+int(STEP_SIZE[1]/2)] = 1
    pos_mask = pos_mask.astype(np.int32)
    return pos_mask


def gen_neg(seg_im):
    r_range = (int(WINDOW_SHAPE[0] / 2) + 1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0] / 2) - 1)
    c_range = (int(WINDOW_SHAPE[1] / 2) + 1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1] / 2) - 1)
    neg_data = list()
    mask = get_pos_mask(seg_im)
    num_points = int(len(seg_im['cellseg']) * 2)
    count = 0
    while(count < num_points):
        rand_cor = ((r_range[1] - r_range[0]) * np.random.random() + r_range[0],
                    (c_range[1] - c_range[0]) * np.random.random() + c_range[0])
        rand_cor = (int(rand_cor[0]), int(rand_cor[1]))
        if(mask[rand_cor[0], rand_cor[1]] != 1):
            window = cut_window(seg_im['imageBF'], rand_cor)
            if((np.max(window)-np.min(window))/np.mean(window) > 0.1):
                neg_data.append(window.reshape(1, WINDOW_SHAPE[0] * WINDOW_SHAPE[1]))
                count += 1
    feat = np.vstack(tuple(neg_data))
    target = np.zeros((feat.shape[0], 1))
    return np.hstack((target, feat))


def gen_neg_example(seg_im_data):
    tables = list()
    idx = 1
    for seg_im in seg_im_data:
        print('processing %s/%s...' %(idx, len(seg_im_data)))
        neg_table = gen_neg(seg_im)
        tables.append(neg_table)
        idx += 1
    return np.vstack(tuple(tables)).astype(np.int32)

"""
seg_im_data = load_seg_im_data('seg_im_data.mat')
neg_data_table = gen_neg_example(seg_im_data)
np.savetxt('negative_not_empty4.csv', neg_data_table, fmt='%d', header = 'negative_examples, first col is target')
"""
"""
id_list = []
for idx in range(Xy.shape[0]):
    im = Xy[idx, 1:]
    if((np.max(im) - np.min(im))/np.mean(im) < 0.1):
        id_list.append(idx)

id_list2 = list(set(list(range(X_neg.shape[0]))), set(id_list))
"""

def get_neg_mask(seg_im):
    pos_mask = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[0]))
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        (r, c) = find_center(seg)
        pos_mask[r-int(STEP_SIZE[0]/2):r+int(STEP_SIZE[0]/2), c-int(STEP_SIZE[1]/2):c+int(STEP_SIZE[1]/2)] = 1
    pos_mask = pos_mask.astype(np.int32)
    neg_mask = binary_dilation(gen_mask(seg_im), np.ones((WINDOW_SHAPE[0],WINDOW_SHAPE[1]))) - pos_mask
    return neg_mask




def gen_pos2(seg_im):
    r_range = (int(WINDOW_SHAPE[0]/2+STEP_SIZE[0]/2)+1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0]/2+STEP_SIZE[0]/2)-1)
    c_range = (int(WINDOW_SHAPE[1]/2+STEP_SIZE[1]/2)+1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1]/2+STEP_SIZE[1]/2)-1)
    pos_data = list()
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        center = find_center(seg)
        if (center[0] > r_range[0] and center[0] < r_range[1] and center[1] > c_range[0] and center[1] < c_range[1]):
            for ii in range(2):
                cor = (STEP_SIZE[0] * np.random.random() - STEP_SIZE[0] / 2, STEP_SIZE[1] * np.random.random() - STEP_SIZE[1] / 2)
                cor = (center[0]+int(cor[0]), center[1]+int(cor[1]))
                window = cut_window(seg_im['imageBF'], cor)
                pos_data.append(window.reshape(1, WINDOW_SHAPE[0] * WINDOW_SHAPE[1]))
    feat = np.vstack(tuple(pos_data))
    target = np.ones((feat.shape[0], 1))
    return np.hstack((target, feat))

def gen_pos_example2(seg_im_data):
    tables = list()
    idx = 1
    for seg_im in seg_im_data:
        pos_table = gen_pos2(seg_im)
        print('processing %s/%s...' %(idx, len(seg_im_data)))
        tables.append(pos_table)
        idx += 1
    return np.vstack(tuple(tables)).astype(np.int32)

"""
seg_im_data = load_seg_im_data('seg_im_data.mat')
pos_data_table = gen_pos_example2(seg_im_data)
np.savetxt('positive_data3.csv', pos_data_table, fmt='%d', header = 'positive_examples3, first col is target')
"""

def gen_neg2(seg_im):
    r_range = (int(WINDOW_SHAPE[0] / 2) + 1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0] / 2) - 1)
    c_range = (int(WINDOW_SHAPE[1] / 2) + 1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1] / 2) - 1)
    neg_data = list()
    neg_mask = get_neg_mask(seg_im)
    num_points = int(len(seg_im['cellseg']) * 3)
    count = 0
    for r in range(r_range[0], r_range[1], STEP_SIZE[0]*3):
        for c in range(c_range[0], c_range[1], STEP_SIZE[1]*3):
            if (neg_mask[r, c] != 1):
                window = cut_window(seg_im['imageBF'], (r, c))
                neg_data.append(window.reshape(1, WINDOW_SHAPE[0] * WINDOW_SHAPE[1]))
    feat = np.vstack(tuple(neg_data))
    target = np.zeros((feat.shape[0], 1))
    return np.hstack((target, feat))


def gen_neg_example2(seg_im_data):
    tables = list()
    idx = 1
    seg_im_data = seg_im_data[0:100]
    for seg_im in seg_im_data:
        print('processing %s/%s...' %(idx, len(seg_im_data)))
        neg_table = gen_neg2(seg_im)
        tables.append(neg_table)
        idx += 1
    return np.vstack(tuple(tables)).astype(np.int32)

"""
seg_im_data = load_seg_im_data('seg_im_data.mat')
neg_data_table = gen_neg_example2(seg_im_data)
np.savetxt('negative_data3.csv', neg_data_table, fmt='%d', header = 'negative_examples2, first col is target')
"""

def get_vertex(seg):
    r1 = np.min(seg[:, 0])
    r2 = np.max(seg[:, 0])
    c1 = np.min(seg[:, 1])
    c2 = np.max(seg[:, 1])
    return (r1, r2, c1, c2)


def gen_rectangle(seg_im):
    r_range = (int(WINDOW_SHAPE[0]/2+STEP_SIZE[0]/2)+1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0]/2+STEP_SIZE[0]/2)-1)
    c_range = (int(WINDOW_SHAPE[1]/2+STEP_SIZE[1]/2)+1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1]/2+STEP_SIZE[1]/2)-1)
    rect_data = list()
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        center = find_center(seg)
        vertex = get_vertex(seg)
        if (center[0] > r_range[0] and center[0] < r_range[1] and center[1] > c_range[0] and center[1] < c_range[1]):
            for ii in range(2):
                cor = (STEP_SIZE[0] * np.random.random() - STEP_SIZE[0] / 2, STEP_SIZE[1] * np.random.random() - STEP_SIZE[1] / 2)
                cor = (center[0]+int(cor[0]), center[1]+int(cor[1]))
                window = cut_window(seg_im['imageBF'], cor)
                new_vertex = np.array([vertex[i]-cor[i//2]+WINDOW_SHAPE[i//2]//2 for i in range(4)]).astype(np.int32)
                data = window.reshape( WINDOW_SHAPE[0] * WINDOW_SHAPE[1])
                line = np.concatenate((new_vertex,data))
                rect_data.append(line)
    return np.vstack(tuple(rect_data))

def gen_rect_example(seg_im_data):
    tables = list()
    idx = 1
    for seg_im in seg_im_data:
        rect_table = gen_rectangle(seg_im)
        print('processing %s/%s...' %(idx, len(seg_im_data)))
        tables.append(rect_table)
        idx += 1
    return np.vstack(tuple(tables)).astype(np.int32)

"""
seg_im_data = load_seg_im_data('seg_im_data.mat')
rect_data_table = gen_rect_example(seg_im_data)
np.savetxt('rect_data.csv', rect_data_table, fmt='%d', header = 'rect_examples, first 4 cols are target')
"""

def gen_mask_data(seg_im):
    r_range = (int(WINDOW_SHAPE[0]/2+STEP_SIZE[0]/2)+1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0]/2+STEP_SIZE[0]/2)-1)
    c_range = (int(WINDOW_SHAPE[1]/2+STEP_SIZE[1]/2)+1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1]/2+STEP_SIZE[1]/2)-1)
    mask_data = list()
    for idx in range(len(seg_im['cellseg'])):
        seg = seg_im['cellseg'][idx, 0]
        center = find_center(seg)
        if (center[0] > r_range[0] and center[0] < r_range[1] and center[1] > c_range[0] and center[1] < c_range[1]):
            for ii in range(2):
                cor = (STEP_SIZE[0] * np.random.random() - STEP_SIZE[0] / 2, STEP_SIZE[1] * np.random.random() - STEP_SIZE[1] / 2)
                cor = (center[0]+int(cor[0]), center[1]+int(cor[1]))
                window = cut_window(seg_im['imageBF'], cor)
                new_seg = (seg - center + np.array([WINDOW_SHAPE[0]//2,WINDOW_SHAPE[1]//2])).astype(np.int32)
                data = window.reshape( WINDOW_SHAPE[0] * WINDOW_SHAPE[1])
                mask = poly2mask(new_seg[:,0], new_seg[:,1], WINDOW_SHAPE).astype(np.int32)
                line = np.concatenate((mask.reshape(WINDOW_SHAPE[0] * WINDOW_SHAPE[1]), data))
                mask_data.append(line)
    return np.vstack(tuple(mask_data))


def gen_mask_example(seg_im_data):
    tables = list()
    idx = 1
    for seg_im in seg_im_data:
        mask_table = gen_mask_data(seg_im)
        print('processing %s/%s...' %(idx, len(seg_im_data)))
        tables.append(mask_table)
        idx += 1
    return np.vstack(tuple(tables)).astype(np.int32)


"""
seg_im_data = load_seg_im_data('seg_im_data.mat')
mask_data_table = gen_mask_example(seg_im_data)
np.savetxt('./data/mask_data.csv', mask_data_table, fmt='%d', header = 'mask_examples, first 2500 cols are target')
"""
