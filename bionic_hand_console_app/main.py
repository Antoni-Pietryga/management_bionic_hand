import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras

from voice_manager import *
from emg_manager import *

voice_model = keras.models.load_model("models/voice_model")
emg_model = keras.models.load_model("models/emg_model")


def voice_manager():
    print("Powiedz komendę")
    record_to_file('voice_recordings/recording.wav')
    files = np.array(tf.io.gfile.glob('voice_recordings/*'))
    print(voice_predict(files, voice_model))


def emg_manager():
    path = input("Podaj ścieżkę pliku do analizy w formacie .npy\n")
    if not os.path.exists(path):
        print("Wybrana ścieżka nie isnieje.")
        return

    data = np.load(path)
    print(frame_predict(data, emg_model))


if __name__ == '__main__':
    while True:
        while True:
            mode = input("Wybierz typ zarządzania: (0-głosem, 1-sygnałami EMG, q-wyjście)\n")
            if mode == 'q':
                break
            elif mode == "0" or mode == "1":
                break
            else:
                print("Niepoprawny typ odpowiedzi.")

        if mode == "0":
            voice_manager()
        elif mode == "1":
            emg_manager()
        else:
            break