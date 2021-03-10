import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
import librosa
import PySimpleGUI as sg

n_fft = 2048
hop_length = 512
sameples_to_consider = 22050  # 1 sec


def get_mfcc(file):
    signal, sr = librosa.load(file)
    if len(signal) > sameples_to_consider:
        signal = signal[:sameples_to_consider]
    mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    mfcc = mfcc.T
    return mfcc


def predict(model, mfcc):
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(mfcc)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    if predicted_index == 0:
        print("I think you are negative")
        sg.popup('I think you are negative')
    else:
        print("I think you are positive, you better visit a doctor to rescan")
        sg.popup('I think you are positive, you better visit a doctor to rescan')


layout = [[sg.Text('Choose audio file to analyze and wait for a minute')],
                 [sg.FileBrowse()],
                 [sg.Submit(), sg.Cancel()]]

window = sg.Window('COVID-19 Coughing Analyzer', layout)

event, values = window.read()
window.close()
mfcc = get_mfcc(values["Browse"])
model = keras.models.load_model("audio_model")
predict(model, mfcc)
