import lstm
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def recover_data(data):
    for i in range(1, len(data))[::-1]:
        data[i] -= data[i - 1]
    return data

def prepare_data(predicted_data, true_data, timeseq, seq_len, origin):
    if origin:
        predicted_data = recover_data(predicted_data)
        true_data = recover_data(true_data)
    predicted_data = predicted_data[seq_len + 1:]
    true_data = true_data[seq_len + 1:]
    timeseq = timeseq[seq_len + 1:]
    for i in range(0, len(predicted_data)):
        if predicted_data[i] < 0:
            predicted_data[i] = true_data[i]
    return predicted_data, true_data, timeseq
    
def plot_results(predicted_data, true_data, timeseq):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(timeseq, true_data, label='True Data', color='blue')
    plt.plot(timeseq, predicted_data, label='Prediction', color='red')
    plt.xlabel("Time(days)")
    plt.ylabel("Popularity")
    plt.title("Hot Trends Prediction")
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def show_mse(predicted_data, true_data):
    predicted_data = np.array(predicted_data)
    true_data = np.array(true_data)
    mse = ((predicted_data - true_data) ** 2).mean(axis=0)
    print("MSE : ", mse, (predicted_data - true_data) ** 2)
    
def get_hot_level(a, b):
    # -2, -1, 0, 1, 2 => ex_down, down, smooth, up, ex_up
    if b == 0:
        return 0
    value = a / b
    if value < 0.1:
        return -2
    elif value < 0.5:
        return -1
    elif value < 2:
        return 0
    elif value < 10:
        return 1
    else:
        return 2

def show_level_result(predicted_data, true_data):
    # Total Accuracy
    right = 0
    for i in range(1, len(predicted_data)):
        a = get_hot_level(predicted_data[i], true_data[i - 1])
        b = get_hot_level(true_data[i], true_data[i - 1])
        if a == b:
            right += 1
    print("Accuracy : %.2f%%" % ((right / (len(predicted_data) - 1)) * 100))
    # Part Accuracy Near Acme
    max_v = max_i = right = 0
    for i in range(0, len(true_data)):
        if true_data[i] > max_v:
            max_v = true_data[i]
            max_i = i
    l = max(1, max_i - 100)
    r = min(len(true_data), max_i + 100)
    for i in range(l, r):
        a = get_hot_level(predicted_data[i], true_data[i - 1])
        b = get_hot_level(true_data[i], true_data[i - 1])
        if a == b:
            right += 1
    print("Accuracy near acme : %.2f%%" % (right / (r - l) * 100))

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 100
    seq_len = 50
    prediction_len = 1

    print('> Loading data... ')

    X_train, y_train, X_test, y_test, data, timeseq = lstm.load_data('example.json', seq_len, True)
    
    #X_train, y_train, X_test, y_test, data, timeseq = lstm.load_event('./event/', seq_len, True)

    print('> Data Loaded. Compiling...')

    if not os.path.exists('lstm_model.h5'):
        model = lstm.build_model_functional([1, 50, 100, 1])
        feature = np.ones(X_train.shape[0])
        model.fit(
            {'main_input' : X_train, 'features_input' : feature},
            y_train,
            batch_size = 32,
            nb_epoch = epochs,
            validation_split=0.05)
        model.save('lstm_model.h5')
        del model
    
    model = load_model('lstm_model.h5')
    
    print('Training duration (s) : ', time.time() - global_start_time)
    
    #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, prediction_len)
    #predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    #predicted = lstm.predict_point_by_point(model, X_test)        
    
    #plot_results_multiple(predictions, y_test, prediction_len)
    #plot_results(predicted, y_test)
    
    predicted_data = lstm.show_prediction_result(model, data, seq_len, prediction_len)
    
    predicted_data, data, timeseq = prepare_data(predicted_data, data, timeseq, seq_len, True)
    
    show_level_result(predicted_data, data)
    
    #predicted_data = lstm.show_prediction_result_multiple(model, data[-54:], seq_len, prediction_len)
    #show_mse(predicted_data, data)
    plot_results(predicted_data, data, timeseq)
    #plot_results_multiple(predicted_data, data[-3:], prediction_len)
    