from util import *
import cv2
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
"""
This file contains functions used to segment the contour of single cell with a seed point.
The principal algorithm is :
1. generate a series of rays of different direnctions from the seed of the cell detected,
2. compute several images which include pix_image, gradx, grady, grad_from_center...
3. get the values of each image on all points on the rays, which is named tab, pixtab, gxtab, gytab...
4. find out the optimal pathway(dynamic programming) through all of the rays which best represents
the cell contour (manually defined scoring function)
5. filter the optimal pathway and get the polygon/mask of the cell
"""
NRAY = 100          # the result is quite sensitive to this parameter
RHO = 30            # the max distance of rays from the center
RHO_SKIP = 5        # the mix distance of rays from the center
AOFF = 1/2/np.pi
MINCELLRADIUS = 5
MAXCELLRADIUS = 50




def findslopes(img):
    """
    usw different kernels to filter the raw_image to get the gradient and 2th gradient image
    """
    img = img.astype(np.float32)
    DY = np.array([[-1,-1,-1],[0, 0, 0],[1, 1, 1]]) * 1/6
    DX = DY.transpose()
    gradx = cv2.filter2D(src=img, ddepth=-1, kernel=DX)
    grady = cv2.filter2D(src=img, ddepth=-1, kernel=DY)

    D2Y = np.array([[0.5, 1, 0.5], [-1, -2, -1], [0.5, 1, 0.5]]) * 0.5
    D2X = D2Y.transpose()
    DXY = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) * 1/4
    grad2x = cv2.filter2D(src=img, ddepth=-1, kernel=D2X)
    grad2y = cv2.filter2D(src=img, ddepth=-1, kernel=D2Y)
    gradxy = cv2.filter2D(src=img, ddepth=-1, kernel=DXY)

    slopes = gradx**2 + grady**2
    slopes2 = grad2x**2 + grad2y**2 + 2 * gradxy**2

    return (slopes, gradx, grady, slopes2, grad2x, grad2y, gradxy)


def get_rays(nray, rho, rho_skip):
    """
    generate a list of rays which start from the seed point
    """
    aoff = 1/2/np.pi
    rays = list()
    for i in range(nray):
        rays.append(np.zeros((rho - rho_skip,2)).astype(np.float32))
        for j in range(rho_skip, RHO):
            [x, y] = pol2cart(2*np.pi*i/nray+aoff, j)
            x = round(x)
            y = round(y)
            rays[i][j - rho_skip, :] = [x, y]
    return rays


def findminpath(tab, gxtab, gytab, pixtab):
    """
    Dynamic programming to findout a optimal pathway through rays which optimize
    the defined scoring function to get the points which best represents the cell
    contour
    """

    pathdist = 2                # the number of points each points on a ray can related to on the previous ray
    pathdist_penalty = 0.3      # penalty of the difference of the pathdist
    pathpix_penalty = 2         # penalty of the difference of pixel values between the point and the previous point
    nray = tab.shape[1]

    #tab = np.hstack((tab,tab[:, 0].reshape(tab.shape[0], 1)))
    #pixtab = np.hstack((pixtab,pixtab[:, 0].reshape(pixtab.shape[0], 1)))
    #gxtab = np.hstack((gxtab,gxtab[:, 0].reshape(gxtab.shape[0], 1)))
    #gytab = np.hstack((gytab,gytab[:, 0].reshape(gytab.shape[0], 1)))

    tab = np.hstack((tab,tab,tab))         # horizontally stack the tab matrix to prepare for the filtering on the result
    pixtab = np.hstack((pixtab,pixtab,pixtab))
    gxtab = np.hstack((gxtab,gxtab,gxtab))
    gytab = np.hstack((gytab,gytab,gytab))

    tab = (tab - tab.min()) / (tab.max() - tab.min())   # noralize the tab matrix
    pixtab = (pixtab - pixtab.min()) / (pixtab.max() - pixtab.min()) * -1       # for we want to find the white contour of the cell so we multipy -1 on the pixtab
    # tab = tab / np.median(tab)
    # pixtab = pixtab / np.median(pixtab)
    path = np.zeros(tab.shape)
    path[:, 0] = np.array(range(0, tab.shape[0]))
    score = np.zeros(tab.shape)
    score[:, 1] = tab[:, 1]

    for i in range(1, tab.shape[1]):
        for j in range(tab.shape[0]):
            mins = np.Inf                   # record the min value of the ray
            minat = 0
            for k in range(-pathdist, pathdist+1):
                if(0 <= (j+k) and (j+k) < tab.shape[0]):
                    s = pixtab[j, i]
                    pixdiff = abs(pixtab[j, i] - pixtab[j+k, i-1])
                    s += pixdiff * pathpix_penalty              # two kinds of penalty
                    s += abs(k) * pathdist_penalty
                    s += score[j+k, i-1]

                    if(s < mins):
                        mins = s
                        minat = j + k
            path[j, i] = minat
            score[j, i]= mins

    start = int(np.argmin(score[:, -1]))
    path = path.astype(np.int32)
    minpath = [start]
    for i in range(tab.shape[1]-1, 0, -1):
        minpath.append(path[minpath[-1], i])
    minpath = minpath[::-1]
    # print(len(minpath))
    minpath = savgol_filter(minpath, 15, 3)             # apply a  Savitzky-Golay filter to the raw minpath signal
    minpath = minpath[nray:nray*2]                      # cut the middle part of minpath whose length is nray
    return np.array(minpath).astype(np.int32)



def get_polygon(img, gradx, grady, seed):
    """
    take raw image, slopes of the image, and the seed point as parameters.
    return the polygon coordinates of the detected cell contour
    """
    rays = get_rays(NRAY, RHO, RHO_SKIP)
    # minCellSize = np.pi * MINCELLRADIUS**2
    # maxCellSize = np.pi * MAXCELLRADIUS**2
    assert 0<seed[0]<img.shape[0] and 0<seed[1]<img.shape[1]
    (cr,cc) = seed                                          # cr, cc is the coordinates of the seed
    [ac, ar] = np.meshgrid(np.array(range(img.shape[0])), np.array(range(img.shape[1])))
    cac = (ac-cc).astype(np.float32)                        # cac,car represent the distance of each pixel on the image to the seed
    car = (ar-cr).astype(np.float32)
    with np.errstate(all='ignore'):
        unitx = np.cos(np.arctan(np.abs(car/cac))) * np.sign(cac)   # unitx,unity represent cosine value of each pixel on the image to the seed
        unity = np.cos(np.arctan(np.abs(cac/car))) * np.sign(car)
        dirslopes = gradx * unitx + grady * unity           # dirslopes is the gradient map which consider the seed points as the center

    tab = np.zeros((RHO - RHO_SKIP, NRAY))
    gxtab = np.zeros((RHO - RHO_SKIP, NRAY))
    gytab = np.zeros((RHO - RHO_SKIP, NRAY))
    pixtab = np.zeros((RHO - RHO_SKIP, NRAY))
    for i in range(NRAY):
        for j in range(RHO-RHO_SKIP):
            pr = int(cr + rays[i][j, 0])
            pc = int(cc + rays[i][j, 1])
            tab[j, i] = dirslopes[pr, pc]
            gxtab[j, i] = gradx[pr, pc]
            gytab[j, i] = grady[pr, pc]
            pixtab[j, i] = img[pr, pc]

    minpath = findminpath(tab, gxtab, gytab, pixtab)        # get the minpath

    polygon = np.zeros((NRAY, 2))
    for i in range(NRAY):
        polygon[i, 0] = cr + rays[i][minpath[i], 0]
        polygon[i, 1] = cc + rays[i][minpath[i], 1]
    #hull = ConvexHull(polygon)
    #polygon = polygon[hull.vertices]
    #print(polygon.shape[0])
    return polygon

def plot_polygon(img, polygon):
    plt.imshow(img, cmap=mpl.cm.gray)
    plt.plot(polygon[:,1], polygon[:, 0], 'r')
    plt.show()

if __name__ == '__main__':
    img = np.array(Image.open('./examples/example1.tif'))
    seed = [90, 300]
    (slopes, gradx, grady, slopes2, grad2x, grad2y, gradxy) = findslopes(img)
    polygon = get_polygon(img, gradx, grady, seed)
    plot_polygon(img, polygon)
