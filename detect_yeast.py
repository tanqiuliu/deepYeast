from keras.models import load_model
from keras.utils import np_utils
from util import *
from segment_seed import *


def cut_window(imageBF, center):
    r1 = int(center[0] - WINDOW_SHAPE[0] / 2)
    r2 = int(center[0] + WINDOW_SHAPE[0] / 2)
    c1 = int(center[1] - WINDOW_SHAPE[1] / 2)
    c2 = int(center[1] + WINDOW_SHAPE[1] / 2)
    return imageBF[r1:r2, c1:c2]

def windows_generator(imageBF, step_size):
    r_range = (int(WINDOW_SHAPE[0] / 2) + 1, IMAGE_SHAPE[0] - int(WINDOW_SHAPE[0] / 2) - 1)
    c_range = (int(WINDOW_SHAPE[1] / 2) + 1, IMAGE_SHAPE[1] - int(WINDOW_SHAPE[1] / 2) - 1)
    for r in range(r_range[0], r_range[1], step_size[0]):
        for c in range(c_range[0], c_range[1], step_size[1]):
            win = cut_window(imageBF, (r, c))
            center = (r, c)
            yield(win, center)


def test_win_std(win):
    return win.std()/win.mean() < 0.1


def test_stripes_std(win):
    r1 = int(WINDOW_SHAPE[0]/3)
    r2 = 2 * int(WINDOW_SHAPE[0]/3)
    c1 = int(WINDOW_SHAPE[1]/3)
    c2 = 2 * int(WINDOW_SHAPE[1]/3)
    if(win[r1:r2, :].std()/win[r1:r2, :].mean() < 0.1 or win[:, c1:c2].std()/win[:, c1:c2].mean() < 0.1):
        return True
    else:
        return False


def judge_yeast(win, model_detect):
    # filter out wrong windows using stdDev/mean within the window, if stdDev/mean<0.1, discard
    if(test_win_std(win)):
        return False
    # same as above, another way to filter out wrong windows
    elif(test_stripes_std(win)):
        return False
    else:
        im = win.reshape(1, WINDOW_SHAPE[0], WINDOW_SHAPE[1], 1)
        result = model_detect.predict(im)
        if(result[0, 0]==0.0 and result[0, 1]==1.0):
            return True
        elif(result[0, 0]==1.0 and result[0, 1]==0.0):
            return False


def get_neighbor_list(center_list, center, neighbor_list):

    pos_up = (center[0]-STEP_SIZE[0], center[1])
    pos_down = (center[0]+STEP_SIZE[0], center[1])
    pos_left = (center[0], center[1]-STEP_SIZE[1])
    pos_right = (center[0], center[1]+STEP_SIZE[1])
    poss = [pos_up, pos_down, pos_left, pos_right]
    # center_list.remove(center)
    neighbor_list.append(center)
    for pos in poss:
        if(pos in center_list and not pos in neighbor_list):
            get_neighbor_list(center_list, pos, neighbor_list)


def get_neighbors(center_list, center):
    neighbors = []
    get_neighbor_list(center_list, center, neighbors)
    return list(set(neighbors))


def merge_multi_detections(center_list):
    for center in center_list:
        center_list1 = center_list[:]
        neighbors = get_neighbors(center_list1, center)
        if(len(neighbors) > 1):
            for n in neighbors:
                center_list.remove(n)
            center_list.append(tuple(np.mean(np.array(neighbors), axis=0).astype(np.int32)))
    return center_list


def detect_centers(imageBF, model_detect):
    center_list = list()
    for (win, center) in windows_generator(imageBF, STEP_SIZE):
        if(judge_yeast(win, model_detect) == True):
            center_list.append(center)
    # center_list = merge_multi_detections(center_list)
    return center_list


def compute_vertex(win, model_rect):
    im = win.reshape(1, WINDOW_SHAPE[0], WINDOW_SHAPE[1], 1)/65535.
    vertex = model_rect.predict(im).astype(np.int32)
    return (vertex[0, 0], vertex[0, 1], vertex[0, 2], vertex[0, 3])


def get_center_list(imageBF, model_detect):
    raw_center_list = detect_centers(imageBF, model_detect)
    center_list = raw_center_list	#postprocessing of centers e.g. merge 
    count = len(raw_center_list)
    new_count = 0

    while(new_count != count):
        count = new_count
        center_list = merge_multi_detections(center_list)
        new_count = len(center_list)

    return center_list


def get_vertex_list(imageBF, model_detect, model_rect):
    center_list = get_center_list(imageBF, model_detect)
    vertex_list = list()
    for center in center_list:
        win = cut_window(imageBF, center)
        vertex = compute_vertex(win, model_rect)
        true_vertex = (vertex[0]+center[0]-WINDOW_SHAPE[0]//2,
                        vertex[1]+center[0]-WINDOW_SHAPE[0]//2,
                        vertex[2]+center[1]-WINDOW_SHAPE[1]//2,
                        vertex[3]+center[1]-WINDOW_SHAPE[1]//2,)
        vertex_list.append(true_vertex)
    return vertex_list


def get_polygon_list(image, center_list):
    (slopes, gradx, grady, slopes2, grad2x, grad2y, gradxy) = findslopes(image)
    polygon_list = list()
    for i in range(len(center_list)):
        print("processing cell %s" %i)
        polygon = get_polygon(image, gradx, grady, center_list[i])
        polygon_list.append(polygon)
    return polygon_list

def plot_detection_center(imageBF, center_list):
    colormap = mpl.cm.gray
    plt.imshow(imageBF, cmap=colormap)
    for center in center_list:
        plt.plot(center[1], center[0], 'r*')
    plt.xlim(0, 512)
    plt.ylim(512, 0)
    plt.show()

def plot_detection_rect(imageBF, vertex_list):
    colormap = mpl.cm.gray
    plt.imshow(imageBF, cmap=colormap)
    for (r1,r2,c1,c2) in vertex_list:
        plt.plot(np.ones(r2-r1)*c1, np.array(range(r1, r2)), 'r')
        plt.plot(np.ones(r2-r1)*c2, np.array(range(r1, r2)), 'r')
        plt.plot(np.array(range(c1, c2)), np.ones(c2-c1)*r1, 'r')
        plt.plot(np.array(range(c1, c2)), np.ones(c2-c1)*r2, 'r')
    plt.xlim(0, IMAGE_SHAPE[1])
    plt.ylim(IMAGE_SHAPE[0], 0)
    plt.show()

def plot_polygons(img, polygon_list):
    plt.imshow(img, cmap=mpl.cm.gray)
    for i in range(len(polygon_list)):
        plt.plot(polygon_list[i][:,1], polygon_list[i][:, 0], 'r')
    plt.xlim(0, IMAGE_SHAPE[1])
    plt.ylim(IMAGE_SHAPE[0], 0)
    plt.show()


if __name__ == '__main__':
    image = np.array(Image.open('./examples/example1.tif'))
    model_detect = load_model('./models/CNN_detect6.h5')
    center_list = get_center_list(image, model_detect)
    polygon_list = get_polygon_list(image, center_list)
    plot_polygons(image, polygon_list)
