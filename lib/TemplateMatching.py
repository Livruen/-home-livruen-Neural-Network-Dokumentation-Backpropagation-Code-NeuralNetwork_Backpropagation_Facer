import cv2
import numpy as np



def templateMatching(grey, template):

    width, height = template.shape[::-1]
    result = cv2.matchTemplate(grey, template, cv2.TM_CCOEFF_NORMED)
    # TemplateWidth, h = cropped.shape[::-1]
    threshold = 0.8
    location = np.where(result >= threshold)
    for point in zip(*location[::-1]):
        cv2.rectangle(grey, point, (point[0] + width, point[1] + height), (0, 0, 255), 2)
