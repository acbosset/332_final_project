import os
import glob
import sys
import cv2


class Stitch:
    def __init__(self):
        pass


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
        img = cv2.imread(i, cv2.IMREAD_COLOR)
        test_images.append(img)

    center_img = length(images) / 2
    print(len(test_images))
    # imS = cv2.resize(test_images[0], (960, 540))
    # cv2.imshow('histogram image', imS)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# sift
# surf
# ransac
