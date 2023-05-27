import cv2
import numpy as np
if __name__ == '__main__':
    multiplied = 30000
    img2 = cv2.imread('/data/Capstone/valid/valid_data/valid_density_gaus/DENSITY_0020.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('/data/Capstone/test/test_data/test_2imgs/IMG_0020.png_sample.png', cv2.IMREAD_GRAYSCALE)
    # height, width = img2.shape
    # raw_width, raw_height = width, height
    # print(img2.shape)
    # img2 = cv2.resize(img2, (256,256))
    img2 = img2/multiplied
    img3 = img3/multiplied
    
    print("GT:", np.sum(img2))
    print("Sample: ", np.sum(img3))
    print("Error: ", np.sum(img2)-np.sum(img3))