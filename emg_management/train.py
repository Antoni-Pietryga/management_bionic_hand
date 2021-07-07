from sklearn.model_selection import train_test_split

from dataset import Dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow import keras


def train():
    patient_data = Dataset().create_dataset()
    patient_data.pop('.\\data')

    key = '.\\data\\s'
    train_keys = [f'{key}{i}' for i in range(1, 11)]
    accuracy_list = []
    for i in range(1, 11):
        test_key = f'{key}{i}'
        test_df = pd.DataFrame(patient_data[test_key], columns=['features', 'labels', 'repetitions'])

        sec_train_df = test_df.loc[(test_df['repetitions'] == 1) | (test_df['repetitions'] == 3)]
        test_df = test_df.loc[(test_df['repetitions'] != 1) & (test_df['repetitions'] != 3)]
        print(test_df.iloc[0]["features"])

        train_keys.remove(test_key)

        train_data = {
            'features': [],
            'labels': [],
            'repetitions': []
        }

        for train_key in train_keys:
            train_data['features'] += patient_data[train_key]['features']
            train_data['labels'] += patient_data[train_key]['labels']
            train_data['repetitions'] += patient_data[train_key]['repetitions']

        train_df = pd.DataFrame(train_data, columns=['features', 'labels', 'repetitions'])

        train_model(train_df, sec_train_df, test_df, i)
        accuracy_list.append(sec_test(test_df, i))
        print(i, "średnia", np.mean(accuracy_list))


def get_model():
    model = Sequential()
    model.add(Dense(64, input_dim=64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(52, activation='softmax'))
    model.compile(loss='categorical_crossentropy',  metrics=['accuracy'], optimizer='adam')
    return model


def train_model(train_df, sec_train_df, test_df, idx):

    X_train = list(train_df['features'])
    y_train = encode_labels(list(train_df['labels']))

    X_pretrain = list(sec_train_df['features'])
    y_pretrain = encode_labels(list(sec_train_df['labels']))

    X_test = list(test_df['features'])
    y_test = encode_labels(list(test_df['labels']))

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)

    model = get_model()
    model.fit(X_train, y_train, epochs=150, verbose=1, validation_data=(X_test, y_test))
    model.save("models/model"+str(idx))
    
    model.evaluate(X_test, y_test)
    model.fit(X_pretrain, y_pretrain, verbose=1, epochs=50)
    model.evaluate(X_test, y_test)
    model.save("models/sec_model" + str(idx))
    sec_test(test_df, idx)


def encode_labels(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    ret = list(np.array(np_utils.to_categorical(encoded_Y), dtype='int'))
    return [list([int(j) for j in i]) for i in ret]


def sec_test(test_df, idx):
    reconstructed_base_model = keras.models.load_model(f'models/model{idx}')
    test_labels = encode_labels(list(test_df["labels"]))
    reconstructed_base_model.evaluate(list(test_df['features']), test_labels)
    reconstructed_model = keras.models.load_model(f'models/sec_model{idx}')
    reconstructed_model.evaluate(list(test_df['features']), test_labels)

    same = 0
    labels = []
    predictions = []
    final_arr = {key:0 for key in range(1, 53)}
    for label in range(1, 53):
        for rep in [2, 4, 5, 6]:
            features = list(test_df.loc[(test_df["labels"]==label) & (test_df["repetitions"]==rep)]['features'])
            result = reconstructed_model.predict(features)
            predicted = []
            pred_ = []
            for i in result:
                if i[np.argmax(i)] > 0.5:
                    predicted.append(np.argmax(i))
                    pred_.append((i[np.argmax(i)], np.argmax(i)+1))
            pred = count_max(pred_)
            labels.append(label)
            predictions.append(pred)
            if label == pred:
                same += 1
                final_arr[label] += 1

    accuracy = float(same)/(52*4)*100
    print(f'result: {accuracy}')
    return accuracy


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


if __name__ == "__main__":
    train()