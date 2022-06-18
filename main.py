import cv2
import numpy as np

def calc_metric(image, x, y, w, h):
    crop_image = image[y:y + h, x:x + w]
    colors, count = np.unique(crop_image.reshape(-1, crop_image.shape[-1]), axis=0, return_counts=True)
    cv2.imshow('crop', crop_image)
    cv2.waitKey(0)
    rect = cv2.rectangle(crop_image, (0, 0), (w, h), (int(colors[count.argmax()][0]), int(colors[count.argmax()][1]), int(colors[count.argmax()][2])), -1)
    cv2.imshow("color", rect)
    cv2.waitKey(0)
    return colors[count.argmax()]

filepath = r"output/1.jpg"
img = cv2.imread(filepath)
a = calc_metric(img, 0, 0, 200, 200)


