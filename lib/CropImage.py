""" This function will crop images """

def crop(image, x, y, width, height):
    """ Crops an image using the specific parameters

    :param image object: image matrix
    :param x point: from which x point in the image
    :param y point: from which y point in the image
    :param width: in the x direction
    :param height: in the y direction
    :return the cropped image
    """
    return image[x:x+width, y:y+height]


