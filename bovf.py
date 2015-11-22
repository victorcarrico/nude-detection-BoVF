# coding: utf-8

import cv
import cv2
import numpy as np

from sklearn.cluster import KMeans


class BoVF(object):
    def __init__(self, images, vocabulary_size=60):
        self.images = images
        self.vocabulary_size = vocabulary_size
        self.descriptors = None

    def _get_descriptor(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv.CV_RGB2GRAY)
        _, des = cv2.SIFT().detectAndCompute(gray, None)

        return des

    def _get_descriptors(self):
        for image in self.images:
            des = self._get_descriptor(image)
            if self.descriptors:
                np.concatenate((self.descriptors, des))
            else:
                self.descriptors = des

    def clusterize(self):
        kmeans = KMeans(n_clusters=self.vocabulary_size)
        kmeans.fit(self.descriptors)

    def histogram(self):
        # TODO
        pass
