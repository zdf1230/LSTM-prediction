import os
import json
import time
import datetime
import warnings
import keras
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.layers import Input, Dense, merge
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def get_list(filename):
    #Read data from json
    input = open(filename, "r", encoding = "UTF-8")
    jsonFileDict = json.loads(input.readline())
    timeSeriesList = jsonFileDict["data"]
    result = []
    timeseq = []
    flag = False
    for i in range(1, len(timeSeriesList)):
        if not flag and timeSeriesList[i][1] > 100:
            t = int(timeSeriesList[i][0]) / 1000
            print(time.strftime('%Y-%m-%d', time.localtime(t)))
            flag = True
        timeSeriesList[i][1] += timeSeriesList[i - 1][1]
        if flag:
            result.append(timeSeriesList[i][1])
            timeseq.append(datetime.datetime.utcfromtimestamp(int(timeSeriesList[i][0]) / 1000))
    return result, timeseq
    
def load_data(filename, seq_len, normalise_window):
    data, timeseq = get_list(filename)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    if normalise_window:
        result = normalise_windows(result)
        
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test, data, timeseq]

def load_event(filepath, seq_len, normalise_window):
    pathDir =  os.listdir(filepath)
    event_d = []
    event_t = []
    max_time = datetime.datetime(1994,12,30)
    for filename in pathDir:
        data, timeseq = get_list(os.path.join('%s%s' % (filepath, filename)))
        event_d.append(data)
        event_t.append(timeseq)
        max_time = max(timeseq[0], max_time)
    for i in range(len(event_d)):
        j = 0
        while event_t[i][j] < max_time:
            j += 1
        event_t[i] = event_t[i][j:]
        event_d[i] = event_d[i][j:]
    event_len = len(event_d[0])
    data = []
    timeseq = []
    for i in range(event_len):
        t = 0
        for j in range(len(event_d)):
            t += event_d[j][i]
        data.append(t / len(event_d))
        timeseq.append(event_t[0][i])

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    if normalise_window:
        result = normalise_windows(result)
        
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test, data, timeseq]    

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data
    
def denormalise(n, p0):
    return (n + 1) * p0

def build_model_sequential(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.1))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))
    
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_functional(layers):
    main_input = Input(shape = (50, layers[0]), name = 'main_input')
    features_input = Input(shape = (1, ), name = 'features_input')
    
    # LSTM Part
    lstm = LSTM(
        input_dim = layers[0],
        output_dim = layers[1],
        return_sequences = True)(main_input)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(
        layers[2],
        return_sequences = False)(lstm)
    lstm = Dropout(0.5)(lstm)
    lstm = Dense(
        output_dim = layers[3],
        activation = 'linear')(lstm)
    
    # Attention Part
    data = merge([lstm, features_input], mode = 'concat')
    attention = Dense(2, activation = 'tanh')(data)
    attention = Activation('softmax')(attention)
    
    # Merge Part
    merged_data = merge([attention, data], mode = 'mul')
    result = Lambda(lambda x: K.sum(x, axis = 1), output_shape=(1, ))(merged_data)
    model = Model(input = [main_input], output = lstm)
    
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model
    
def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
    
def show_prediction_result(model, data, seq_len, prediction_len):
    #Show original data comparation
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = normalise_windows(result)
    result = np.array(result)
    result = np.reshape(result, (result.shape[0], result.shape[1], 1))
    predicted = model.predict({'main_input' : result[:, :-1], 'features_input' : np.zeros(result.shape[0])})
    while prediction_len > 1:
        for i in range(len(result)):
            tmp = np.append(result[i][1:], predicted[i])
            result[i] = np.reshape(tmp, (tmp.shape[0], 1))
        predicted = model.predict({'main_input' : result[:, :-1], 'features_input' : np.zeros(result.shape[0])})
        prediction_len -= 1
    predicted_data = data[0:sequence_length]
    for i in range(len(predicted)):
        predicted_data.append(denormalise(predicted[i][0], data[i]))
    return predicted_data
    
def show_prediction_result_multiple(model, data, seq_len, prediction_len):
    #Show original data comparation by predicting multiple days
    prediction_seqs = []
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = normalise_windows(result)
    result = np.array(result)
    result = np.reshape(result, (result.shape[0], result.shape[1], 1))
    
    for i in range(int(len(result) / prediction_len)):
        curr_frame = result[i * prediction_len]
        predicted = []
        predicted_data = []
        for j in range(prediction_len):
            t = model.predict(curr_frame[newaxis, :, :])[0,0]
            predicted.append(t)
            predicted_data.append(denormalise(t, data[i * prediction_len]))
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [seq_len-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted_data)
    return prediction_seqs