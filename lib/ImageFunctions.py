import numpy as np
import cv2
import CropImage


window_name = 'This window'
gray_scale = cv2.COLOR_BGR2GRAY


def turnImageToGray(image_path):
    """ Changes every image to a gray image

    :param image_path:
    :return gray image:
    """
    img_obj = loadImage(image_path)
    return cv2.cvtColor(img_obj, gray_scale)  # change image to gray


def loadImage(image_path):
    """
    :param image_path:
    :return image object:
    """
    return cv2.imread(image_path, 1)


def saveImage(image, newImageName):
    """saves the gray image as gray_imageName.imagetype

    :param image object:
    :return:
    """
    cv2.imwrite(newImageName, image)


def display(image):
    """ Displays the Image in a Window with specific name
    :param image:
    :return:
    """
    cv2.imshow(window_name, image)

    cv2.waitKey(0) # Waits for a pressed key
    cv2.destroyAllWindows()



