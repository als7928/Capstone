import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)
if __name__=="__main__":
    simg = cv2.imread('/data/Capstone/test/test_data/test_2imgs/IMG_0052.png_sample.png', cv2.IMREAD_GRAYSCALE)
    simg = cv2.resize(simg, (1024, 1024))
    _, img = cv2.threshold(simg, 10, 255, cv2.THRESH_BINARY)
    erosion = cv2.erode(img, kernel, iterations= 1)
    th, img_thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    simg = cv2.cvtColor(simg, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(contours)):
        cv2.drawContours(simg, [contours[i]], 0, (255,0,0), 2)

    cv2.imwrite('counted.png',simg)
    print(len(contours))