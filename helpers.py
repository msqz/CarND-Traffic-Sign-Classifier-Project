import numpy as np
from scipy import ndimage
import cv2
import random
import math
from tqdm import tqdm
from multiprocessing import Pool


def normalize(data):
    normalized = np.array(data, dtype=np.int16)
    return (normalized - 128) / 128


def augment(dataset):
    augmented = []

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
            augmented.append(resized)
        # every 3rd image gets blurred
        elif i % 3 == 0:
            sigma = random.randrange(3, 11, 2)
            augmented.append(ndimage.gaussian_filter(img, sigma=sigma))
        # others get sharpened
        else:
            f = ndimage.gaussian_filter(img, sigma=1)
            augmented.append(2 * img - f)

    return augmented


def equalize_part(data, label, top):
    count = len(data)
    missing = top - count
    if missing == 0:
        return data, np.full(len(data), label)

    if missing <= count:
        addition = data[:missing]
    else:
        addition = np.repeat(
            data, missing//count + 1, axis=0)[:missing]

    equalized_data = np.concatenate((data, augment(addition)))
    return equalized_data, np.full(len(equalized_data), label)


def equalize(data, labels):
    frequency = np.array(np.unique(labels, return_counts=True)).T
    top = np.max(frequency[:, 1])

    buckets = {}
    for label in frequency[:, 0]:
        indices = np.where(labels == label)
        buckets[label] = data[indices]

    with Pool(12) as p:
        partials = p.starmap(equalize_part, zip(
            buckets.values(), buckets.keys(), np.full(len(frequency), top)))

    data_expanded = np.concatenate([p[0] for p in partials])
    labels_expanded = np.concatenate([p[1] for p in partials])

    return data_expanded, labels_expanded
