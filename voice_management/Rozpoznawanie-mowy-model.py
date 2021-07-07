#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import random

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
data_dir = pathlib.Path('datasets/po_normalizacji')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
filenames_czynnosciowy = np.array(tf.io.gfile.glob(str(data_dir)+'/czynnosciowy' + '/*'))
filenames_szczypcowy = np.array(tf.io.gfile.glob(f'{str(data_dir)}/szczypcowy' + '/*'))
filenames_pensetowy = np.array(tf.io.gfile.glob(f'{str(data_dir)}/pensetowy' + '/*'))
filenames_pozycja_neutralna = np.array(tf.io.gfile.glob(f'{str(data_dir)}/pozycjaneutralna' + '/*'))
filenames_wieszakowy = np.array(tf.io.gfile.glob(f'{str(data_dir)}/wieszakowy' + '/*'))
filenames_zaciskowy = np.array(tf.io.gfile.glob(f'{str(data_dir)}/zaciskowy' + '/*'))
#filenames = tf.random.shuffle(filenames)
#num_samples = len(filenames)
kfold = KFold(5)
count = 0
for train, test in kfold.split(filenames_czynnosciowy):
    valid = train[-5:]
    train = train[:-5]
    print(train)
    print(valid)
    print(test)
    train_files = np.concatenate([filenames_czynnosciowy[train],filenames_szczypcowy[train],filenames_pozycja_neutralna[train],filenames_pensetowy[train],filenames_wieszakowy[train],filenames_zaciskowy[train]])
    val_files = np.concatenate([filenames_czynnosciowy[valid],filenames_szczypcowy[valid],filenames_pozycja_neutralna[valid],filenames_pensetowy[valid],filenames_wieszakowy[valid],filenames_zaciskowy[valid]])
    test_files = np.concatenate([filenames_czynnosciowy[test],filenames_szczypcowy[test],filenames_pozycja_neutralna[test],filenames_pensetowy[test],filenames_wieszakowy[test],filenames_zaciskowy[test]])


    def decode_audio(audio_binary):
      audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
      return tf.squeeze(audio, axis=-1)

    def get_label(file_path):
      parts = tf.strings.split(file_path, os.path.sep)
      return parts[-2]

    def get_waveform_and_label(file_path):
      label = get_label(file_path)
      audio_binary = tf.io.read_file(file_path)
      waveform = decode_audio(audio_binary)
      return waveform, label

    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    def get_spectrogram(waveform):
      # Padding for files with less than 16000 samples
      size=int(tf.shape(waveform))+1
      zero_padding = tf.zeros([160000] - tf.shape(waveform), dtype=tf.float32)

      # Concatenate audio with padding so that all audio clips will be of the 
      # same length
      waveform = tf.cast(waveform, tf.float32)
      equal_length = tf.concat([waveform, zero_padding], 0)
      spectrogram = tf.signal.stft(
          equal_length, frame_length=255, frame_step=128)

      spectrogram = tf.abs(spectrogram)

      return spectrogram
    for waveform, label in waveform_ds.take(1):
      label = label.numpy().decode('utf-8')
      spectrogram = get_spectrogram(waveform)

    def get_spectrogram_and_label_id(audio, label):
      spectrogram = get_spectrogram(audio)
      spectrogram = tf.expand_dims(spectrogram, -1)
      label_id = tf.argmax(label == commands)
      return spectrogram, label_id
    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    def preprocess_dataset(files):
      files_ds = tf.data.Dataset.from_tensor_slices(files)
      output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
      output_ds = output_ds.map(
          get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
      return output_ds
    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    for spectrogram, _ in spectrogram_ds.take(1):
      input_shape = spectrogram.shape
    num_labels = len(commands)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32), 
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    EPOCHS = 200
    history = model.fit(
        train_ds, 
        validation_data=val_ds,  
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=30, monitor="val_loss",restore_best_weights=True),
    )
    model.save(f"model_{count}")
    count += 1
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

