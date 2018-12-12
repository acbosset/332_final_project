import cv2
import numpy as np
import os
import glob
import sys
import scipy.misc
import matplotlib.pyplot as plt
from random import randrange
import sift as mysift
try:
    image_file = sys.argv[1]
except:
    image_file = "test_images"


def stitch(img1, img2, gimg1, gimg2):
    surf = cv2.xfeatures2d.SURF_create()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = mysift.start(gimg1)
    kp2, des2 = mysift.start(gimg2)
    # kp1, des1 = surf.detectAndCompute(gimg1, None)
    # kp2, des2 = surf.detectAndCompute(gimg2, None)
    print len(kp1), 'kp1 feature points'
    print len(kp2), 'kp1 feature points'
    kp_img1 = cv2.drawKeypoints(
        gimg1, kp1, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img2 = cv2.drawKeypoints(
        gimg2, kp2, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('my_sift_keypoints_1.jpg', kp_img1)
    cv2.imwrite('my_sift_keypoints_2.jpg', kp_img2)

    # sift.compute()  gets descriptors based off

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print len(matches)
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)
    print "matches", len(matches)
    # for results
    # print len(matches)
    # img_matches = cv2.drawMatchesKnn()
    # cv2.imwrite('sift_matches_knn.jpg', img_matches)
    # img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], outImg=np.array([]), flags=2)
    # cv2.imwrite('sift_matches_reg.jpg', img_match)
    # edit this
    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        # print src
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        print "whoopy"
        print "finds homography"
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    print H

    val = img2.shape[1] + img1.shape[1]
    # img1 = img1.astype(np.float32)
    # img2 = img2.astype(np.float32)
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
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    reg_images.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_images.append(img)

result_image = stitch(reg_images[1], reg_images[0], test_images[1], test_images[0])

cv2.imshow('dst image', result_image)
cv2.waitKey(0)
cv2.imwrite("my_stitch_output_myroom.jpg", result_image)
cv2.destroyAllWindows()
