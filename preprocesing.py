import nibabel as nib
from nibabel.testing import data_path
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import cv2

class preprocessing:

    def __init__(self) -> None:
        pass
    

    def get_diff_img(self, test_load, verbose = 1):
        '''
        input: test load, single nii image dim(112,112,25,100)
        return: array of diff img dim(25,112,112)
        '''
        if verbose:
            plt.figure() 
        norm = np.zeros((112,112))
        norm2 = np.zeros((112,112))
        norm3 = np.zeros((112,112))
        result = []
        for slice in range(25):
            diff_sum = np.zeros((112,112))
            for i in range(99):
                cv2.normalize(test_load[:,:,slice,i],  norm) 
                cv2.normalize(test_load[:,:,slice,i+1],  norm2) 
                (score, diff) = compare_ssim(norm , norm2, full=True)
                diff_sum =+ diff
            diff_sum = 1. - diff_sum
            cv2.normalize(diff_sum, norm3,norm_type=cv2.NORM_MINMAX)
            if verbose:   
                plt.subplot(5,5,slice+1),plt.imshow(norm3)
                plt.title('diff'), plt.xticks([]), plt.yticks([])
            result.append(np.copy(norm3))
        return result

    def apply_threshold(self, test_load, threshold = 0.01, verbose = 1):
        '''
        input: array of diff img dim(25,112,112)
        return: array of diff img thresholded dim(25,112,112)
        '''
        if verbose:
            plt.figure()
        result = []
        for i, slice in enumerate(test_load):
            slice[slice<threshold] = 0
            result.append(slice)
            if verbose:
                plt.subplot(5,5,i+1),plt.imshow(slice)
                plt.title('threshold'), plt.xticks([]), plt.yticks([])
        return result


    def run(self, test_load):
        step1 = self.get_diff_img(test_load)
        step2 = self.apply_threshold(step1)

        plt.show()
        return step2
        

preprocess = preprocessing()
    