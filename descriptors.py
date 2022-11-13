import numpy as np
import cv2 as cv

class Descriptors:

    '''
    1. calculate diff img
    2. find points of interest
    3. find POI that are on every image
    4. calculate LBP in that POI
    5. use it as feature vec
    '''


    def __init__(self) -> None:
        self.sift = cv.SIFT_create()


    def detect(self, img_slices):
        keypoints = [], descriptors = []
        for img_slice in img_slices:
            keypoints_temp, descriptors_temp = self.sift.detectAndCompute(img_slice,None)
            keypoints.append(keypoints_temp)
            descriptors.append(descriptors_temp)
        return keypoints, descriptors

    
descriptors = Descriptors()