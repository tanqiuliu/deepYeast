from util import *
from skimage.morphology import disk

def background_screening(img):
img = skimage.img_as_ubyte(img)
max_img = skimage.filter.rank.maximum(img, disk(3))
min_img = skimage.filter.rank.minimum(img, disk(3))
mean_img = skimage.filter.rank.mean(img, disk(3))
bg = (max_img - min_img)/mean_img < 0.1
