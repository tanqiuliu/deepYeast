import scipy.io as sio

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
