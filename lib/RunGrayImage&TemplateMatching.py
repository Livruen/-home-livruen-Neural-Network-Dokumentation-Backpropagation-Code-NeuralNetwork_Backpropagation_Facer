import CropImage
import AreaTable
from ImageFunctions import *
from TemplateMatching import templateMatching

if __name__ == '__main__':

    image_name = 'images/pic.jpeg'

    grey = turnImageToGray(image_name)
    saveImage(grey, "images/grey_" + image_name)

    # # How to crop the Image ?
    x = input('Define x')
    y = input('Define y')
    TemplateWidth = input('Define width')
    TemplateHeight = input('Define height')

    template = CropImage.crop(grey, x, y, TemplateWidth, TemplateHeight)
    display(template)

    templateMatching(grey, template)
    cv2.imwrite('images/res.png', grey)
    display(grey)
    print AreaTable.compute_summed_area_table(grey)
