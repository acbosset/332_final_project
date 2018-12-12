import os
import glob
import sys
import cv2
import scipy.misc
import numpy as np
import copy
import math


def scale_space(img):
    blur = img
    octaves = {}
    for o in range(4):
        octaves[o] = []
        next_pic = cv2.resize(blur, (0, 0), fx=0.5, fy=0.5)
        for i in range(5):
            blur = cv2.GaussianBlur(blur, (5, 5), 2)
            # cv2.imshow('histogram image', blur)
            # cv2.waitKey(0)
            octaves[o].append(blur)
        blur = next_pic
    return octaves


def d_o_g(octaves):
    dog = {}
    for x in range(4):
        dog[x] = []
        blurred_images = octaves[x]
        for i in range(len(blurred_images) - 1):
            dog[x].append(cv2.subtract(blurred_images[i], blurred_images[i + 1]))

    return dog

# maybe filter here for the intensity


def maxima(dog):  # returns list of lists
    maxima_images_octaves = []
    octave_keypoints = []
    for i in range(len(dog)):
        print "octave ", i
        octave = dog[i]
        maxima_imgs = []
        keypoints = []
        for img_num in range(1, len(octave) - 1):
            img = copy.deepcopy(octave[img_num])
            maximas = np.zeros((img.shape[0], img.shape[1]))
            # img = cv2.cornerHarris(img, 2, 3, 0.04)
            # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1, None)
            ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_TOZERO)
            print "shape[0]:", img.shape[0]
            print "shape[1]", img.shape[1]
            for x in range(1, img.shape[0] - 1):
                # print"x", x
                for j in range(1, img.shape[1] - 1):
                    # print "y", j
                    max_pix = True
                    pix = img[x][j]
                    # if x > img.shape[1]:
                    #     print "pixel value", pix
                    # just comparing to pixel
                    for new_pic in range(-1, 2):
                        comp_pix = octave[img_num + new_pic]
                        for xp in range(3):
                            for yp in range(3):
                                if pix < comp_pix[x - 1 + xp][j - 1 + yp]:
                                    # if x > img.shape[1]:
                                        # print "pixel value", pix
                                        # print "comp pix", comp_pix[x - 1 + xp][j - 1 + yp]
                                    max_pix = False
                                    break
                            if not max_pix:
                                break
                        if not max_pix:
                            break
                    if max_pix and img[x][j] != 0:
                        # if x > img.shape[1]:
                        keypoints.append((j, x))
                        maximas[x][j] = 255

            maxima_imgs.append(img)
        octave_keypoints.append(list(set(keypoints)))
        # add a
        maxima_images_octaves.append(maxima_imgs)
        print "length of keypoints for each octave", len(keypoints)
    print "length of octave_keypoints", len(octave_keypoints)
    return maxima_images_octaves, octave_keypoints


def linapprox(img, x, y):
    return img[x][y] + (img[x][y + 1] - img[x][y - 1]) * 2 + (img[x + 1][y] - img[x - 1][y]) * 2


def magnitude(img, x, y):
    return math.sqrt(math.pow((linapprox(img, x + 1, y) - linapprox(img, x - 1, y)), 2) + math.pow((linapprox(img, x, y + 1) - linapprox(img, x, y - 1)), 2))


def orientation(img, x, y):
    return math.atan((linapprox(img, x, y + 1) - linapprox(img, x, y - 1)) / (linapprox(img, x + 1, y) - linapprox(img, x - 1, y)))

# could pass a list of keypoint locations or just an image with keypoints and an original intensity image


def image_gradient(img):

    gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # Define kernel for y differences
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    x = cv2.filter2D(img, cv2.CV_32F, kernel=gx)
    y = cv2.filter2D(img, cv2.CV_32F, kernel=gy)
    mags = np.hypot(x, y)
    thetas = np.rad2deg(np.arctan2(y, x))
    return mags, thetas


def create_keypoints(keypoints_octaves, img):
    octave_num = 0
    #img = cv2.GaussianBlur(img, (5, 5), 1.5 * 2)
    first_mags, thetas = image_gradient(img)
    kps = []
    for keypoints in keypoints_octaves:
        new_sigma = 2 ** (octave_num) * 1.5
        mags = cv2.GaussianBlur(first_mags, (5, 5), new_sigma)
        window_size = int(5 * new_sigma)
        wind = int(window_size / 2)
        for idx in keypoints:
            idx_histogram = [0] * 36
            key_x = idx[0]
            key_y = idx[1]
            for x in range(key_x - wind, key_x + wind + 1):
                for y in range(key_y - wind, key_y + wind + 1):
                    if x < img.shape[0] and y < img.shape[1]:
                        m = mags[x][y]
                        o = thetas[x][y]
                        bin = int(math.floor(o / 10)) + 18
                        idx_histogram[bin] = idx_histogram[bin] + m

            hist = idx_histogram
            angle = idx_histogram.index(max(idx_histogram)) * 10 + 5
            hist_max = max(idx_histogram)
            # del hist[hist.index(hist_max)]
            kp = cv2.KeyPoint(key_x, key_y, window_size, _angle=angle, _octave=3 - octave_num)
            kps.append(kp)
            # while(max(hist) / float(hist_max) > .8 and hist_max != 0):
            #     print hist
            #     hist_val = hist.pop(hist.index(max(hist)))
            #     angle = idx_histogram.index(hist_val)
            #     kp = cv2.KeyPoint(x, y, window_size, _angle=angle, _octave=octave_num)
        octave_num = octave_num + 1
    print "feature points selected", (len(kps))
    return kps

# for testing sift algorithm
# if __name__ == '__main__':
#     try:
#         image_file = sys.argv[1]
#     except:
#         image_file = "test_images"
#
#     testing_dir = os.getcwd() + "/" + image_file
#     test_path = os.path.join(testing_dir, '*')
#     images = glob.glob(test_path)
#     gray_images = []
#
#     for i in images:
#         img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
#         gray_images.append(img)
#
#     octs = scale_space(gray_images[0])
#     dogs = d_o_g(octs)
#     keypoint_imgs, keypoints = maxima(dogs)
#     kps = create_keypoints(keypoints, gray_images[0])
#     sift = cv2.xfeatures2d.SIFT_create()
#     kps, des = sift.compute(gray_images[0],  kps)
    # i = 0
    # for im in octs[1]:
    #     cv2.imshow('scale_space', im)
    #     # cv2.imwrite("scale space" + str(i) + ".jpg", im)
    #     cv2.waitKey(0)
    #     i = i + 1
    #
    # i = 0
    # for im in dogs[1]:
    #     cv2.imshow('gauss diff', im)
    #     # cv2.imwrite("gaussian_diff_" + str(i) + ".jpg", im)
    #     cv2.waitKey(0)
    #     i = i + 1
    # i = 0
    # for im in dst_octaves[1]:
    #     cv2.imshow('dst image', im)
    #     # cv2.imwrite("maxima_corners" + str(i) + ".jpg", im)
    #     cv2.waitKey(0)
    #     i = i + 1


def start(img):
    # img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)

    octs = scale_space(img)
    dogs = d_o_g(octs)
    keypoint_imgs, keypoints = maxima(dogs)

    kps = create_keypoints(keypoints, img)
    sift = cv2.xfeatures2d.SIFT_create()
    # img = cv2.GaussianBlur(img, (5, 5), 3)
    kps, des = sift.compute(img,  kps)
    return kps, des
