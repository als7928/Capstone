import numpy as np
import math
import cv2
import mmcv
import scipy.io as sio

def GaussianKernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian kernel which is equal to MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    radius_x, radius_y = [(radius-1.)/2. for radius in shape]
    y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
    h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumofh = h.sum()
    if sumofh != 0:
        h /= sumofh
    return h


# def create_dmap(img, gtLocation, depth, beta=0.25, downscale=8.0):
#     width, height = img.size
#     raw_width, raw_height = width, height
#     width = math.floor(width / downscale)
#     height = math.floor(height / downscale)
#     raw_loc = gtLocation
#     gtLocation = gtLocation / downscale
#     gaussRange = 25
#     # kernel = GaussianKernel(shape=(25, 25), sigma=3)
#     pad = int((gaussRange - 1) / 2)
#     densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
#     for gtidx in range(gtLocation.shape[0]):
#         if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
#             xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
#             yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
#             x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
#             x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)
#             y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
#             y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
#             depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
#             kernel = GaussianKernel((25, 25), sigma=beta * 5 / depth_mean)
#             densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
#     densityMap = densityMap[pad:pad + height, pad:pad + width]
#     return densityMap

def create_dmap(img, gtLocation, depth, sigma, downscale=1.0):
    height, width, cns = img.shape
    raw_width, raw_height = width, height
    width = math.floor(width / downscale)
    height = math.floor(height / downscale)
    raw_loc = gtLocation
    gtLocation = gtLocation / downscale
    gaussRange = 25
    pad = int((gaussRange - 1) / 2)
    densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
    for gtidx in range(gtLocation.shape[0]):
        if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
            xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
            yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
            x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
            x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)

            y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
            y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
            depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
            # print(np.sum(depth[y_down:y_up]), np.sum(depth[x_down:x_up]))
            # print(depth_mean)
            if depth_mean != 0:
                kernel = GaussianKernel((25, 25), sigma=sigma / depth_mean)
            else:
                kernel = GaussianKernel((25, 25), sigma=sigma)
            densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
    densityMap = densityMap[pad:pad + height, pad:pad + width]
    return densityMap


def load_depth():
    depth = sio.loadmat("DEPTH_1009.mat")
    depth = depth['depth']
    depth[depth > 20000] = 0
    # depth[depth > 255] = 0
    depth[depth < 0] = 0
    depth = depth / 20000
    depth = depth * 255
    depth = depth.astype(np.uint8)
    # depth = depth / 255
    cv2.imshow("dd", depth)
    cv2.waitKey()
    return depth

def load_point():
    loc = sio.loadmat("GT_1009.mat")
    loc = loc['point'].astype(np.float32)
    return loc
    

if __name__ == '__main__':
        img = mmcv.imread("IMG_1009.png")
        # load annotation
        # annot is a numpy array (N, 5).
        # N means there are N objects in the image, 5 means {x1,y1,x2,y2,class_id}

        depth = load_depth()
        # print(depth)

        loc = load_point()
        dmap = create_dmap(img, loc, depth, 0.6, downscale=4.0)
        new_image = dmap.astype(np.uint8)
        new_rgb = np.dstack([dmap, dmap, dmap])
        print(dmap)
        # print(new_rgb)
        out = 1*img 
        img = cv2.resize(img, (480, 270))
        # cv2.imshow("dd", new_rgb*255)
        # cv2.imshow("dd3", img)
        # cv2.imwrite("dd.png", new_rgb*255)
        # cv2.waitKey()