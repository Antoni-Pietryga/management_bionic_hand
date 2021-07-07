import scipy.io as sio
import os
from tqdm import tqdm
import numpy as np
import json

from utils import E2_NAME, E3_NAME, E1_CLASSES, E2_CLASSES, INCCORECRT_LABEL, WINDOW_SIZE


class Dataset:

    def __init__(self):
        self.patients_raw_data = {}
        self.patients_data = {}
        self.data_path = os.path.join(".", "data")

    def load_raw_data(self):
        for subdir, dirs, files in os.walk(self.data_path):
            patient = subdir.split('/')[-1]
            self.patients_raw_data[patient] = {}

            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(subdir, file)

                    exercise = file.split('.')[0].split('_')[1]
                    self.patients_raw_data[patient][exercise] = sio.loadmat(file_path)

    def create_dataset(self):
        if not os.path.exists(os.path.join("processed_data", "patient_data.json")):
            self.load_raw_data()

            for patient in tqdm(self.patients_raw_data.keys()):
                self.patients_data[patient] = {"features": [], "labels": [], "repetitions": []}
                for exercise in self.patients_raw_data[patient].keys():
                    self.process_exercise(patient, exercise)

            self.save_dataset()
            return self.patients_data

        else:
            return self.load_processed_data()

    def process_exercise(self, patient, exercise):
        exercise_data = self.patients_raw_data[patient][exercise]
        emg = exercise_data["emg"]
        restimulus = exercise_data["restimulus"]
        rerepetition = exercise_data["rerepetition"]

        data_size = emg.shape[0]
        window_iterator = 0

        while window_iterator + WINDOW_SIZE < data_size:

            window_label = int(restimulus[window_iterator][0])

            same_class_count = 0
            while same_class_count < WINDOW_SIZE and restimulus[window_iterator + same_class_count][0] == window_label:
                same_class_count += 1

            if same_class_count == WINDOW_SIZE:

                if window_label == INCCORECRT_LABEL:
                    window_iterator += same_class_count
                    continue

                window_label = self.rename_label(window_label, exercise)
                window_emg = emg[window_iterator:window_iterator+WINDOW_SIZE]
                window_feat = self.time_statistics(window_emg)
                window_repetition = rerepetition[window_iterator][0]

                self.patients_data[patient]["labels"].append(window_label)
                self.patients_data[patient]["features"].append(window_feat)
                self.patients_data[patient]["repetitions"].append(int(window_repetition))

                window_iterator += int(WINDOW_SIZE/2)
            else:
                window_iterator += same_class_count

    @staticmethod
    def rename_label(window_label, exercise):
        if window_label != INCCORECRT_LABEL:
            if exercise == E3_NAME:
                window_label += E1_CLASSES + E2_CLASSES
            elif exercise == E2_NAME:
                window_label += E1_CLASSES
        return window_label

    def save_dataset(self):
        if not os.path.exists("processed_data"):
            os.makedirs("processed_data")
        with open(os.path.join("processed_data", "patient_data.json"), "w") as json_file:
            json.dump(self.patients_data, json_file)

    @staticmethod
    def load_processed_data():
        with open(os.path.join("processed_data", "patient_data.json"), "r") as json_file:
            return json.load(json_file)

    @staticmethod
    def time_statistics(raw):
        window_size = raw.shape[0]
        num_channels = raw.shape[1]
        threshold = 2

        num_zeros = np.zeros(num_channels, dtype=np.uint16)
        num_slope_changes = np.zeros(num_channels, dtype=np.uint16)
        waveform_length = np.zeros(num_channels, dtype=np.uint16)

        mean_abs = np.mean(np.abs(raw), axis=0, dtype=np.float_)

        for i in range(num_channels):
            for j in range(window_size):

                if (j > 0) and (j < window_size - 1):

                    # Zero crossing
                    prev = raw[j - 1][i]
                    curr = raw[j][i]
                    next_ = raw[j + 1][i]

                    if ((curr > 0 and next_ < 0) or (curr < 0 and next_ > 0)) and np.abs(curr - next_) >= threshold:
                        num_zeros[i] += 1

                    # Slope sign changes
                    condition_1 = (curr > prev) and (curr > next_)
                    condition_2 = (curr < prev) and (curr < next_)
                    condition_3 = np.abs(curr - next_) >= threshold or np.abs(curr - prev) >= threshold

                    if (condition_2 or condition_1) and condition_3:
                        num_slope_changes[i] += 1

                    # Waveform Length
                    if j > 0:
                        waveform_length[i] += np.abs(curr - prev)

        time_stat_vec = list(np.concatenate((mean_abs, num_zeros, num_slope_changes, waveform_length)))

        return time_stat_vec


if __name__ == "__main__":
    dataset = Dataset()
    patients_data = dataset.create_dataset()
