import numpy as np
import keras

from emg_manager import *

emg_model = keras.models.load_model("models/emg_model")

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

frame = np.load('example_emg_frames/emg_frame2.npy')
outout_frames = []

for i in range(0, len(frame), 100):
    if i+200 < len(frame):
        outout_frames.append(time_statistics(frame[i:i+200]))

result = emg_model.predict(outout_frames)
pred_ = []
for i in result:
    if i[np.argmax(i)] > 0.5:
        pred_.append((i[np.argmax(i)], np.argmax(i) + 1))

pred = count_max(pred_)

print(pred)