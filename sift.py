import os
import glob
import sys
import cv2
import scipy.misc
import numpy as np
import copy
import math


class Stitch:
    def __init__(self):
        pass

# sift step 1


def scale_space(img):
    blur = img
    octaves = {}
    for o in range(4):
        octaves[o] = []
        next_pic = cv2.resize(blur, (0, 0), fx=0.5, fy=0.5)
        for i in range(5):
            blur = cv2.GaussianBlur(blur, (5, 5), 3) # CHANGE for window size if not 3
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


def maxima(dog):  # returns list of lists
    maxima_octaves = []
    for i in range(len(dog)):
        # print "octave ", i
        octave = dog[i]
        maxima_img = []
        for img_num in range(1, len(octave) - 1):
            img = copy.deepcopy(octave[img_num])
            for x in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    max_pix = True
                    pix = img[x][j]
                    # just comparing to pixel
                    for new_pic in range(-1, 2):
                        comp_pix = octave[img_num + new_pic]
                        for xp in range(3):
                            for yp in range(3):
                                if pix < comp_pix[x - 1 + xp][j - 1 + yp]:
                                    img[x][j] = 0
                                    max_pix = False
                                    break
                            if not max_pix:
                                break
                        if not max_pix:
                            break
                    if max_pix and img[x][j] != 0:
                        img[x][j] = 255
            maxima_img.append(img)
        maxima_octaves.append(maxima_img)
        print(maxima_octaves)
    return maxima_octaves

# NEED TO CONSOL INTO ONE IMAGE OF KEYPOINTS WITH INTENSITIES OF OTHER PIXELS

def linapprox(img, x, y):
    return img[x][y] + (img[x][y+1] - img[x][y-1])*2 + (img[x+1][y] - img[x-1][y])*2

def magnitude(img, x, y):
    return math.sqrt(math.pow((linapprox(img, x+1, y) - linapprox(img, x-1, y)), 2) + math.pow((linapprox(img, x, y+1) - linapprox(img, x, y-1)), 2))

def orientation(img, x, y):
    return math.atan((linapprox(img, x, y+1) - linapprox(img, x, y-1)) / (linapprox(img, x+1, y) - linapprox(img, x-1, y)))

# could pass a list of keypoint locations or just an image with keypoints and an original intensity image
def keypointorientations(img, img_keypts):
    # find keypoints
    np.histogarm(
    for key in keypoints:
        # get key x,y
        for x in range(key_x - 1,key_x + 2):
            for y in range(key_y - 1, key_y + 2):
                m = magnitude(img, x, y)
                o = orientation(img, x, y)
                 # add mag to histogram in bin corr to orientation


# window size = 5x5 * 1.5*simga
# go to each keypoint, look at a window of 5x5 * 1.5sigma and calculate orientation and mag for each
# make histogram of mag * orientation, 36 bins for degrees
# check if any other peaks @ at least 80% of highest peak
# assign orientation to bin #





if __name__ == '__main__':
    try:
        image_file = sys.argv[1]
    except:
        image_file = "test_images"

    testing_dir = os.getcwd() + "/" + image_file
    test_path = os.path.join(testing_dir, '*')
    images = glob.glob(test_path)
    test_images = []
    for i in images:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
        test_images.append(img)

    octs = scale_space(test_images[0])
    dogs = d_o_g(octs)
    max_octaves = maxima(dogs)

    dst_octaves = []
    for i in range(len(max_octaves)):
        octave = max_octaves[i]
        dst_imgs = []
        #print "range octave", len(octave)
        for img_num in range(len(octave)):
            img = copy.deepcopy(octave[img_num])
            dst = cv2.cornerHarris(img, 2, 3, 0.04)
            ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
            dst_imgs.append(dst)
        dst_octaves.append(dst_imgs)
    # print dst_octaves.shape

    i = 0
    for im in octs[1]:
        cv2.imshow('scale_space', im)
        # cv2.imwrite("scale space" + str(i) + ".jpg", im)
        cv2.waitKey(0)
        i = i + 1

    i = 0
    for im in dogs[1]:
        cv2.imshow('gauss diff', im)
        # cv2.imwrite("gaussian_diff_" + str(i) + ".jpg", im)
        cv2.waitKey(0)
        i = i + 1
    i = 0
    for im in dst_octaves[1]:
        cv2.imshow('dst image', im)
        # cv2.imwrite("maxima_corners" + str(i) + ".jpg", im)
        cv2.waitKey(0)
        i = i + 1

    cv2.destroyAllWindows()
