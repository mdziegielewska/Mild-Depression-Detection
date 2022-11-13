import nibabel as nib
from nibabel.testing import data_path
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import cv2
from preprocesing import preprocess
from descriptors import descriptors

class Scheduler:

    def __init__(self) -> None:
        self.path = 'C:\\Users\\Acer\\Desktop\\szkola\\wizja\\ds002748\\'


    def load_data(self):
        files = []
        for i in range(72):
            index = f'0{i+1}' if i<9 else i
            files.append(nib.load(f'{self.path}sub-{index}\\func\\sub-{index}_task-rest_bold.nii.gz').get_fdata())
        return files


    def run(self):
        for file in self.load_data():
            preprocessed = preprocess.run(file)

