import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import warnings

warnings.filterwarnings("ignore")


class preprocessing:

    def __init__(self) -> None:
        pass

    def get_diff_img(self, test_load, verbose=False):
        """
        input: test load, single nii image dim(112,112,25,100)
        return: array of diff img dim(25,112,112)
        """
        if verbose:
            plt.figure()

        norm = np.zeros((112, 112))
        norm2 = np.zeros((112, 112))
        norm3 = np.zeros((112, 112))
        result = []

        for slice in range(25):
            diff_sum = np.zeros((112, 112))

            for i in range(99):
                cv2.normalize(test_load[:, :, slice, i], norm)
                cv2.normalize(test_load[:, :, slice, i + 1], norm2)
                (score, diff) = compare_ssim(norm, norm2, full=True)
                diff_sum = + diff
            diff_sum = 1. - diff_sum
            cv2.normalize(diff_sum, norm3, norm_type=cv2.NORM_MINMAX)

            if verbose:
                plt.subplot(5, 5, slice + 1), plt.imshow(norm3)
                plt.title('diff'), plt.xticks([]), plt.yticks([])
            result.append(np.copy(norm3))

        return result

    def apply_threshold(self, test_load, threshold=0.0001, verbose=False):
        """
        input: array of diff img dim(25,112,112)
        return: array of diff img thresholded dim(25,112,112)
        """
        if verbose:
            plt.figure()
        result = []

        for i, slice in enumerate(test_load):
            slice[slice < threshold] = 0
            # result.append(np.copy(slice))

            if verbose:
                plt.subplot(5, 5, i + 1), plt.imshow(slice)
                plt.title('threshold'), plt.xticks([]), plt.yticks([])

        return test_load

    def convert_type(self, test_load, verbose=False):
        if verbose:
            plt.figure()
        img_list = [(img * 255).astype(np.uint8) for img in test_load]

        if verbose:
            for i, slice in enumerate(img_list):
                plt.subplot(5, 5, i + 1), plt.imshow(slice)
                plt.title('UINT8'), plt.xticks([]), plt.yticks([])

        return img_list

    def save_preprocessed_data(self, test_load, patient, depressed=True):
        img_list = [(img * 255).astype(np.uint8) for img in test_load]

        image_path = f"C:/Users/martae/Desktop/repos/Mild-Depression-Detection/preprocessed_data/"

        if depressed:
            image_path = os.path.join(image_path, "depression/")
        else:
            image_path = os.path.join(image_path, "no_depression/")

        if not os.path.exists(image_path):
            os.mkdir(image_path)

        patient_path = os.path.join(image_path, f"patient_{patient}")

        if not os.path.exists(patient_path):
            os.mkdir(patient_path)

        for i, slice in enumerate(img_list):
            print(f"saving slice {i+1}")
            cv2.imwrite(f"{patient_path}/patient_{patient}-slice_{i+1}.png", slice)

    def run(self, test_load, patient=None, depressed=False, verbose=False):
        step1 = self.get_diff_img(test_load, verbose)
        step2 = self.apply_threshold(step1, threshold=0.001, verbose=verbose)
        # self.save_preprocessed_data(step2, patient, depressed=depressed)
        step3 = self.convert_type(step2, verbose=verbose)
        plt.show()

        return step3


preprocess = preprocessing()
