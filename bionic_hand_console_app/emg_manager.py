import numpy as np
import json
from gestures.gestures_dict import gestures_dict


def count_max(predictions):
    if len(predictions) == 0:
        return -1
    count_pred = {i: [] for i in range(53)}
    for prob, label in predictions:
        count_pred[label].append(prob)
    sorted_pred = dict(sorted(count_pred.items(), key=lambda item: len(item[1])))
    elements = []

    for key in sorted_pred:
        if len(sorted_pred[key]) == len(sorted_pred[list(sorted_pred.keys())[-1]]):
            elements.append([sorted_pred[key], key])

    if len(elements) == 1:
        return elements[0][1]
    elif len(elements) == 0:
        return -1
    else:
        max_ = 0
        max_idx = 0
        idx = 0
        for e in elements:
            e[0] = np.mean(e[0])
            if e[0] > max_:
                max_ = e[0]
                max_idx = idx
            idx += 1
        return elements[max_idx][1]


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


def frame_predict(frame, emg_model):
    outout_frames = []

    for i in range(0, len(frame), 100):
        if i + 200 < len(frame):
            outout_frames.append(time_statistics(frame[i:i + 200]))

    result = emg_model.predict(outout_frames)
    pred_ = []
    for i in result:
        if i[np.argmax(i)] > 0.5:
            pred_.append((i[np.argmax(i)], np.argmax(i) + 1))

    pred = count_max(pred_)
    gesture_configuration = gestures_dict[str(pred)]
    if gesture_configuration != "unknown":
        return json.load(open(f"gestures/{gesture_configuration}"))
    return None
