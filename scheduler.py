from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import cv2
from preprocesing import preprocess
from descriptors import descriptors
from threading import Lock

score_sick = 0
score_healthy = 0
processed_count = 0
lock = Lock()


class Scheduler:

    def __init__(self) -> None:
        # self.path = 'C:\\Users\\Acer\\Desktop\\szkola\\wizja\\ds002748\\'
        self.path = 'C:/Users/martae/Desktop/repos/Mild-Depression-Detection/dataset/'

    def load_data(self):
        for i in range(72):
            index = f'0{i + 1}' if i < 9 else i + 1
            # filepath = f'{self.path}sub-{index}\\func\\sub-{index}_task-rest_bold.nii.gz'
            filepath = f'{self.path}sub-{index}_func_sub-{index}_task-rest_bold.nii.gz'
            file = nib.load(filepath).get_fdata()
            print(f'{((i + 1) * 100 / 72):.2f}% loaded {filepath}')
            yield file

    def preprocess_data(self, data):
        for patient, file in enumerate(data):
            if patient < 51:
                isDepressed = True
            else:
                isDepressed = False

            patient = f'0{patient + 1}' if patient < 9 else patient + 1
            print("Preprocessing for patient:", patient)
            preprocess.run(file, patient, depressed=isDepressed, verbose=False)

    def predict(self, patient, file, predict_descriptors, predict_index):
        global lock
        score_sick_local = 0
        score_healthy_local = 0

        if int(predict_index) != patient+1:
            preprocessed = preprocess.run(file, verbose=False)
            keypoints_tmp, descriptors_tmp = descriptors.detect(preprocessed)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            for slice_idx in range(25):
                if descriptors_tmp[slice_idx] is not None and predict_descriptors[slice_idx] is not None:
                    if len(predict_descriptors[slice_idx]) >= 2 and len(descriptors_tmp[slice_idx]) >= 2:
                        matches = flann.knnMatch(predict_descriptors[slice_idx], descriptors_tmp[slice_idx], k=2)
                        matches = [x for x in matches if len(x) > 1]
                        matches = sorted(matches, key=lambda x: (x[0].distance + x[1].distance))
                        min_match_count = 3

                        if len(matches) >= min_match_count:
                            score = np.sum(
                                [(match[0].distance + match[1].distance) for match in matches[0:min_match_count]])
                        else:
                            score = 70 * min_match_count

                        if patient < 51:
                            score_sick_local += score
                        else:
                            score_healthy_local += score

            with lock:
                global score_sick, score_healthy, processed_count
                score_sick += score_sick_local
                score_healthy += score_healthy_local
                processed_count += 1
                # print(f'score_sick: {score_sick}, score_healthy: {score_healthy} ')
                print(f'{(processed_count * 100 / 72):.2f}% processed')

    def run(self):
        keypoints = []
        descriptors2 = []
        keypoints_tmp = []
        descriptors_tmp = []

        data_generator = self.load_data()

        # to preprocess and save data
        # self.preprocess_data(data_generator)

        index = '70'
        # filepath = f'{self.path}sub-{index}\\func\\sub-{index}_task-rest_bold.nii.gz'
        filepath = f'{self.path}sub-{index}_func_sub-{index}_task-rest_bold.nii.gz'

        predict_patient = nib.load(filepath).get_fdata()
        predict_preprocessed = preprocess.run(predict_patient, verbose=False)
        predict_keypoints, predict_descriptors = descriptors.detect(predict_preprocessed)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for patient, file in enumerate(data_generator):
                futures.append(executor.submit(self.predict, patient, file, predict_descriptors, index))
            for future in futures:
                future.result()

        global score_sick, score_healthy
        print(score_sick)
        print(score_healthy)
        score_sick /= (51 if int(index) >= 51 else 50)
        score_healthy /= (20 if int(index) >= 51 else 21)
        predict = score_sick / (score_sick + score_healthy)
        print(f"{predict * 100}% of score is \"healthy\" score. Subject probably do{' ' if predict <= 0.5 else ' not '}"
              f"have mild depression!")

        #     draw_params = dict(matchColor = (0,255,0),
        #     singlePointColor = (255,0,0),
        #     #matchesMask = matchesMask,
        #     flags = cv2.DrawMatchesFlags_DEFAULT)

        # img3 = cv2.drawMatchesKnn(preprocessed_list[0][slice_idx],keypoints[0][slice_idx],preprocessed_list[1]
        # [slice_idx],keypoints[1][slice_idx],
        #                         matches, None, **draw_params)
        # plt.imshow(img3),plt.show()


if __name__ == "__main__":
    scheduler = Scheduler()
    scheduler.run()
