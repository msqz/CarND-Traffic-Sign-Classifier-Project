import numpy as np
from scipy import ndimage
import cv2
import random
import math


def randomize(dataset):
    randomized = []

    for i, img in enumerate(dataset):
        # every 2nd image gets rotated
        if i % 2 == 0:
            rotation = random.randrange(-15, 16, 3)
            rotated = ndimage.rotate(img, rotation)
            crop_h = int(
                (img.shape[0]/2)/math.tan(math.radians(90-abs(rotation)))) + 1
            crop_w = int(
                (img.shape[1]/2)/math.tan(math.radians(90-abs(rotation)))) + 1
            cropped = rotated[crop_w:-crop_w, crop_h:-crop_h]
            resized = cv2.resize(cropped, (img.shape[1], img.shape[0]))
            randomized.append(resized)
        # every 3rd image gets blurred
        elif i % 3 == 0:
            sigma = random.randrange(3, 11, 2)
            randomized.append(ndimage.gaussian_filter(img, sigma=sigma))
        # others get sharpened
        else:
            f = ndimage.gaussian_filter(img, sigma=1)
            randomized.append(2 * img - f)

    return randomized


def fill_up(data, labels):
    data_expanded = np.copy(data)
    labels_expanded = np.copy(labels)
    frequency = np.array(np.unique(labels, return_counts=True)).T

    buckets = {}
    for label in frequency[:, 0]:
        indices = np.where(labels == label)
        buckets[label] = data[indices]

    top = np.max(frequency[:, 1])

    for f in frequency:
        label = f[0]
        count = f[1]
        missing = top - count
        if missing == 0:
            continue

        if missing <= count:
            addition = buckets[label][:missing]
        else:
            addition = np.repeat(
                buckets[label], missing//count + 1, axis=0)[:missing]

        addition = randomize(addition)
        data_expanded = np.concatenate((data_expanded, addition))
        labels_expanded = np.concatenate(
            (labels_expanded, np.full((missing,), label)))

    return data_expanded, labels_expanded
