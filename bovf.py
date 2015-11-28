# coding: utf-8

import cv
import cv2
import numpy as np

from sklearn.cluster import KMeans


class BoVF(object):
    def __init__(self, labeled_images, vocabulary_size=60):
        self.images = labeled_images
        self.vocabulary_size = vocabulary_size
        self.descriptors = None
        # self.classified_descriptors = {}
        self.image_descriptors = []
        self.histogram_list = []

    def _get_descriptor(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv.CV_RGB2GRAY)
        _, des = cv2.SIFT().detectAndCompute(gray, None)

        return des

    def _get_descriptors(self):
        for image, label in self.images.iteritems():
            des = self._get_descriptor('dbNudeDetection/{}'.format(image))
            self.image_descriptors.append(({image: label}, des, {}))
            if self.descriptors is not None:
                self.descriptors = np.concatenate((self.descriptors, des))
            else:
                self.descriptors = des

    def clusterize(self):
        self.kmeans = KMeans(n_clusters=self.vocabulary_size)
        self.kmeans.fit(self.descriptors)
        # for descriptor in self.descriptors:
        #     self.classified_descriptors.update(
        #         {descriptor: self.kmeans.predict(descriptor)})

    def histogram(self):
        for im in self.image_descriptors:
            for descriptor in im[1]:
                cluster = self.kmeans.predict(descriptor)
                occur = im[2].get(cluster[0])
                im[2].update({cluster[0]: occur + 1})
