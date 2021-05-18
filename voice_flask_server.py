import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import os
import time
import numpy as np
import librosa
import json
import scipy
from scipy import io
from pyvad import trim
from flask import Flask, jsonify, request, Response, make_response
from collections import OrderedDict
from scipy.fftpack import dst, dct
from Custom_layer import *

# For Check Time
bb = time.time()

# For Flask API
app = Flask(__name__)

# GPU Setting
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)
K.set_session(session)
K.clear_session()

# Load Model
model_E_crop = load_model('Model_Emotion_crop.h5')     # Emotion model from melspectrogram
model_G_crop = load_model('Model_Gender_crop.h5')      # Gender model
model_A_crop = load_model('Model_Age_crop.h5')         # Age model
model_E_feature = load_model('Model_Emotion_Features.h5', custom_objects={'Spatial_multiply': Spatial_multiply, 'Channel_multiply': Channel_multiply})     # Emotion model from speech features
model_E_combine = load_model('Model_Emotion_Merge.h5')     # Emotion model

model_E_crop._make_predict_function()
model_G_crop._make_predict_function()
model_A_crop._make_predict_function()
model_E_feature._make_predict_function()
model_E_combine._make_predict_function()

model_E_combine.summary()
# For Check Time2
start_time = time.time()

# Pre-processing
Num_Frame = 1500    # max wave length (15 sec)
Stride = 0.01       # stride (10ms)
Window_size = 0.025  # filter window size (25ms)
Num_data = 1
Num_mels = 40       # Mel filter number
pre_emphasis = 0.97  # Pre-Emphasis filter coefficient
Num_Crop_Frame = 200  # Frame size of crop

bin_start = 0   # Histogram bin interval start
bin_end = 10    # Histogram bin interval end
N_bins = 40     # Histogram number of bins
bins = np.arange(bin_start, bin_end + (bin_end - bin_start) / N_bins, (bin_end - bin_start) / N_bins)       # Histogram bins
Num_moments = 8        # Moments order
DCT_DST_clip = 40    # DCT or DST clip size


def preprocessing(y, sr):
    # Resampling to 16kHz
    if sr != 16000:
        sr_re = 16000  # sampling rate of resampling
        y = librosa.resample(y, sr, sr_re)
        sr = sr_re

    # Denoising
    y[np.argwhere(y == 0)] = 1e-10
    y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)

    # Pre Emphasis filter
    y_Emphasis = np.append(y_denoise[0], y_denoise[1:] - pre_emphasis * y_denoise[:-1])

    # Normalization (Peak)
    y_max = max(y_Emphasis)
    y_Emphasis = y_Emphasis / y_max  # normalize for VAD

    # Voice Activity Detection (VAD)
    vad_mode = 2  # VAD mode = 0 ~ 3
    y_vad = trim(y_Emphasis, sr, vad_mode=vad_mode, thr=0.01)   # trim using VAD module
    if y_vad is None:
        y_vad = y_Emphasis

    # De normalization
    y_vad = y_vad * y_max

    # Obtain the mel spectrogram
    S = librosa.feature.melspectrogram(y=y_vad, sr=sr, hop_length=int(sr * Stride), n_fft=int(sr * Window_size), n_mels=Num_mels, power=2.0)
    r, Frame_length = S.shape

    # Obtain the normalized mel spectrogram
    S_norm = (S - np.mean(S)) / np.std(S)

    # zero padding
    Input_Mels = np.zeros((r, Num_Frame), dtype=float)
    if Frame_length < Num_Frame:
        Input_Mels[:, :Frame_length] = S_norm[:, :Frame_length]
    else:
        Input_Mels[:, :Num_Frame] = S_norm[:, :Num_Frame]

    # Obtain the log mel spectrogram
    w = 1e+6
    S_mel_log = np.log(1 + w * S)

    # Feature
    Input_DCT, Input_DST = Feature_DCT_DST(S_mel_log)
    Input_DCT = np.expand_dims(np.expand_dims(Input_DCT, axis=0), axis=-1)
    Input_DST = np.expand_dims(np.expand_dims(Input_DST, axis=0), axis=-1)
    Input_Hist = np.expand_dims(np.expand_dims(Feature_Hist(S_mel_log), axis=0), axis=-1)
    Input_Moments = np.expand_dims(np.expand_dims(Feature_Moments(S_mel_log), axis=0), axis=-1)

    return Input_Mels, Input_DCT, Input_DST, Input_Hist, Input_Moments, Frame_length


def Crop_Mels(Input_Mels_origin, Each_Frame_Num):
    Input_Mels_origin = Input_Mels_origin.T

    Crop_stride = int(Num_Crop_Frame / 2)

    # Calculate the number of cropped mel-spectrogram
    if Each_Frame_Num > Num_Frame:
        Number_of_Crop = int(Num_Frame / Crop_stride) - 1
    else:
        if Each_Frame_Num < Num_Crop_Frame:
            Number_of_Crop = 1
        else:
            Number_of_Crop = int(round(Each_Frame_Num / Crop_stride)) - 1

    # Crop

    Cropped_Mels = np.zeros((Number_of_Crop, Num_Crop_Frame, Input_Mels_origin.shape[1]))
    crop_num = 0  # the number of cropped data
    if Each_Frame_Num > Num_Frame:  # If the frame number is higher than 1500, the number of crop is 14
        Each_Crop_Num = int(Num_Frame / Crop_stride) - 1
        for N_crop in range(0, Each_Crop_Num):
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * Crop_stride:N_crop * Crop_stride + Num_Crop_Frame, :]
            crop_num += 1
    else:
        if Each_Frame_Num < Num_Crop_Frame:    # If the frame number is lower than 200, the number of crop is 1
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[:Num_Crop_Frame, :]
            crop_num += 1
        else:
            Each_Crop_Num = int(round(Each_Frame_Num / Crop_stride)) - 1    # Calculate the number of crop
            if round(Each_Frame_Num / Crop_stride) < Each_Frame_Num / Crop_stride:
                for N_crop in range(0, Each_Crop_Num):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * Crop_stride:N_crop * Crop_stride + Num_Crop_Frame, :]
                    crop_num += 1
            else:
                for N_crop in range(0, Each_Crop_Num - 1):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * Crop_stride:N_crop * Crop_stride + Num_Crop_Frame, :]
                    crop_num += 1
                shift_frame = int((Each_Frame_Num / Crop_stride - round(Each_Frame_Num / Crop_stride)) * Crop_stride)
                Cropped_Mels[crop_num, :, :] = Input_Mels_origin[(Each_Crop_Num - 1) * Crop_stride + shift_frame:(Each_Crop_Num - 1) * Crop_stride + shift_frame + Num_Crop_Frame, :]
                crop_num += 1

    return Cropped_Mels, Number_of_Crop


def Feature_DCT_DST(log_S):
    r2, Frame_length = log_S.shape

    # DCT, DST in time direction
    DCT_Mels_log_temp = dct(log_S, axis=1) / Frame_length
    DST_Mels_log_temp = dst(log_S, axis=1) / Frame_length

    if Frame_length < DCT_DST_clip:
        DCT_Mels_log = np.zeros((r2, DCT_DST_clip))
        DST_Mels_log = np.zeros((r2, DCT_DST_clip))
        DCT_Mels_log[:, :Frame_length] = DCT_Mels_log_temp
        DST_Mels_log[:, :Frame_length] = DST_Mels_log_temp
    else:
        DCT_Mels_log = DCT_Mels_log_temp[:, :DCT_DST_clip]
        DST_Mels_log = DST_Mels_log_temp[:, :DCT_DST_clip]

    return DCT_Mels_log, DST_Mels_log


def Feature_Hist(log_S):
    Hist_Mels_log = np.zeros((log_S.shape[0], N_bins))

    for ii in range(0, log_S.shape[0]):
        Hist_Mels_log[ii, :], temp = np.histogram(log_S[ii, :], bins=bins, density=True)

    return Hist_Mels_log


def Feature_Moments(log_S):
    r2, Frame_length = log_S.shape

    # Normalized moments
    Moments_Mels_log = np.zeros((r2, Num_moments), dtype=float)

    # Mean
    Mean_Mels_log = np.mean(log_S, axis=1)
    Moments_Mels_log[:, 0] = Mean_Mels_log
    Mean_Mels_log = np.expand_dims(Mean_Mels_log, axis=1) * np.ones((1, log_S.shape[1]))

    # Std
    Std_Mels_log = np.std(log_S, axis=1)
    Moments_Mels_log[:, 1] = Std_Mels_log

    # Normalized moments
    for n_moments in range(3, Num_moments + 1):
        Moment_Mels_log_temp = np.mean((log_S - Mean_Mels_log) ** n_moments, axis=1)
        Moments_Mels_log[:, n_moments-1] = np.where(Moment_Mels_log_temp > 0, 1, -1) * np.abs(Moment_Mels_log_temp) ** (1 / n_moments) / Std_Mels_log

    return Moments_Mels_log


# Main Code
def generate(path_name, wav_name):

    # File Dir = Load from client.py(json)
    audio_name = wav_name
    audio_path = path_name

    y, sr = librosa.load(audio_path + audio_name)
    # Preprocessing(Resampling, Normalization, Denoising, Pre-emphasis, VAD)
    Input_Mels, Input_DCT, Input_DST, Input_Hist, Input_Moments, Frame_length = preprocessing(y, sr)

    # Crop mel-spectrogram
    Cropped_Mels, Number_of_Crop = Crop_Mels(Input_Mels, Frame_length)
    Cropped_Mels = np.reshape(Cropped_Mels, (Cropped_Mels.shape[0], Cropped_Mels.shape[1], Cropped_Mels.shape[2], 1))

    """Emotion"""
    # Predict from cropped log melsectrogram
    y_E_pred_crop = np.mean(model_E_crop.predict(Cropped_Mels), axis=0)  # emotion
    y_E_pred_crop = np.expand_dims(y_E_pred_crop, axis=0)

    # Predict Emotion from speech features
    y_E_pred_dct, y_E_pred_dst, y_E_pred_hist, y_E_pred_moments, y_E_pred_total = model_E_feature.predict([Input_DCT, Input_DST, Input_Hist, Input_Moments])

    # Predict emotion
    y_E_pred = model_E_combine.predict([y_E_pred_crop, y_E_pred_dct, y_E_pred_dst, y_E_pred_hist, y_E_pred_moments, y_E_pred_total])

    """Age"""
    # Predict from cropped log melsectrogram
    y_A_pred_crop = np.mean(model_A_crop.predict(Cropped_Mels), axis=0)  # Age

    # Age Denormalize
    mean_age, std_age = 33.33587, 7.69647       # Pre-defined mean and std of age data
    y_A_pred = y_A_pred_crop * std_age + mean_age    # Age prediction value

    # Age group
    y_A_pred_group = np.zeros((3,))

    if y_A_pred >= 40:  # 40s
        y_A_pred_group[2] = 1
    elif y_A_pred >= 30:  # 30s
        y_A_pred_group[1] = 1
    elif y_A_pred >= 20:  # 20s
        y_A_pred_group[0] = 1

    """Gender"""
    # Predict from cropped log melsectrogram
    y_G_pred = np.mean(model_G_crop.predict(Cropped_Mels), axis=0)  # Gender

    # Ready for data
    result = OrderedDict()

    # Emotion (Happiness:10001, Anger:10002, Disgust:10003, Fear:10004, Neutral:10005, Sadness:10006, Surprise:10007)
    result["10001"] = round(float(y_E_pred[0, 3]), 4)
    result["10002"] = round(float(y_E_pred[0, 0]), 4)
    result["10003"] = round(float(y_E_pred[0, 1]), 4)
    result["10004"] = round(float(y_E_pred[0, 2]), 4)
    result["10005"] = round(float(y_E_pred[0, 4]), 4)
    result["10006"] = round(float(y_E_pred[0, 5]), 4)
    result["10007"] = round(float(y_E_pred[0, 6]), 4)

    # Age (Predicted Age:20000, 20s:20003, 30s:20004, 40s:20005)
    result["20000"] = round(float(y_A_pred), 4)
    result["20003"] = round(float(y_A_pred_group[0]), 4)
    result["20004"] = round(float(y_A_pred_group[1]), 4)
    result["20005"] = round(float(y_A_pred_group[2]), 4)

    # Gender (Male:30001, Female:30002)
    result["30001"] = round(float(y_G_pred[0]), 4)
    result["30002"] = round(1-float(y_G_pred[0]), 4)

    return result


@app.route('/', methods=['POST', 'GET'])
def delivered_json():
    print('request.form : {}'.format(request.form))
    path_name = request.form['path_dir']
    wav_name = request.form['wav_name']
    aa = time.time()
    result = generate(path_name, wav_name)
    print('it takes {:.2f}s'.format(time.time() - aa))
    return json.dumps(result)

if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - bb))
    app.run(host='0.0.0.0', port=9999)