import cv2
import numpy as np
import os
import glob
import sys
import scipy.misc
import matplotlib.pyplot as plt
from random import randrange

try:
    image_file = sys.argv[1]
except:
    image_file = "test_images"


def stitch(img1, img2, gimg1, gimg2):
    #sift = cv2.xfeatures2d.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gimg1, None)
    kp2, des2 = sift.detectAndCompute(gimg2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    # print H

    val = img2.shape[1] + img1.shape[1]
    result_image = cv2.warpPerspective(img1, H, (val, img2.shape[0]))
    result_image[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result_image


testing_dir = os.getcwd() + "/" + image_file
test_path = os.path.join(testing_dir, '*')
images = glob.glob(test_path)
reg_images = []
test_images = []
for i in images:
    img = cv2.imread(i)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    reg_images.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_images.append(img)

result_image = stitch(reg_images[1], reg_images[0], test_images[1], test_images[0])
bw_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
# three_result_image = stitch(reg_images[2], result_image, test_images[2], bw_result)
# three_result_image = cv2.resize(three_result_image, (0, 0), fx=0.3, fy=0.3)
cv2.imshow('dst image', result_image)
# cv2.imshow('three image', three_result_image)
cv2.waitKey(0)
cv2.imwrite("stitch_output_myroom.jpg", result_image)
cv2.destroyAllWindows()
