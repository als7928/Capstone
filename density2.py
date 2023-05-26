import cv2
import numpy as np
if __name__ == '__main__':
    img2 = cv2.imread('/data/Capstone/valid/valid_data/valid_density1200/DENSITY_0000.png', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('05-14_Clipimg_Crossattn(L2)(540epoch)_sample.png', cv2.IMREAD_GRAYSCALE)
    # height, width = img2.shape
    # raw_width, raw_height = width, height
    # print(img2.shape)
    # img2 = cv2.resize(img2, (256,256))
    img2 = img2/1200
    
    print(np.sum(img2))