import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)
if __name__=="__main__":
    simg = cv2.imread('05-14_Clipimg_Crossattn(L2)(540epoch)_sample.png', cv2.IMREAD_GRAYSCALE)
    simg = cv2.resize(simg, (1024, 1024))
    _, img = cv2.threshold(simg, 10, 255, cv2.THRESH_BINARY)
    med = cv2.medianBlur(img, 3)
    erosion = cv2.erode(img, kernel, iterations= 1)
    # cv2.imshow('erosion', erosion)
    th, img_thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,2)

    dst2 = cv2.cvtColor(simg, cv2.COLOR_GRAY2BGR)


    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(dst2, [contours[i]], 0, (255, 0, 255), 2)

    print(len(contours))
    #cv2.imshow('th2',th2)
    #cv2.imshow('simg', simg)

    #cv2.imshow('dst', dst2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()