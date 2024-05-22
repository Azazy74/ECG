import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

def preprocess_ecg_data(data_path):
    ecg_signals = []
    ecg_fields = []
    filtered_signals = []

    for subject in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject)
        for file in os.listdir(subject_path):
            file_path = os.path.join(subject_path, file)
            print(file_path)
            signal, fields = wfdb.rdsamp(file_path.split('.')[0], channels=[1])
            ecg_signals.append(signal)
            ecg_fields.append(fields)
            break

    print(len(ecg_signals[0]))
    print(ecg_fields[0])

    # Preprocessing
    ecg_corrected = []
    # Define bandpass filter parameters
    low_cut = 1.0
    high_cut = 40.0
    order = 2

    # Apply bandpass filter to the signal
    for i in range(len(ecg_signals)):
        fs = ecg_fields[i]['fs']
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, ecg_signals[i][:, 0])
        filtered_signals.append(filtered)

    # Apply a high-pass filter to remove baseline wander and DC drift
    for i in range(len(filtered_signals)):
        b, a = butter(2, 0.5 / (1000 / 2), 'highpass')
        ecg_filtered = filtfilt(b, a, filtered_signals[i])
        ecg_corrected.append(ecg_filtered)

    # Estimate the isoelectric line (baseline) using a moving average filter
    window_size = int(1000 * 0.15)  # 150 ms window size
    baseline = []
    for i in range(len(ecg_corrected)):
        baseline.append(savgol_filter(ecg_corrected[i], window_size, 1))

    # Subtract the estimated baseline from the filtered ECG signal
    for i in range(len(ecg_corrected)):
        filtered_signals[i] = ecg_corrected[i] - baseline[i]

    # Differentiation
    diff_signals = []
    for i in range(len(filtered_signals)):
        diff_signal = np.diff(filtered_signals[i])
        diff_signals.append(diff_signal)

    # Squaring
    squared_signals = []
    for i in range(len(diff_signals)):
        squared_signal = diff_signals[i] ** 2
        squared_signals.append(squared_signal)

    # Moving-window integration
    integrated_signals = []
    window_size = int(0.15 * fs)  # 150 ms window
    for i in range(len(squared_signals)):
        integrated_signal = np.convolve(squared_signals[i], np.ones(window_size)/window_size, mode='same')
        integrated_signals.append(integrated_signal)

    # Thresholding
    thresholded_signals = []
    threshold = 0.5 # This value should be adjusted based on the data
    for i in range(len(integrated_signals)):
        thresholded_signal = np.where(integrated_signals[i] > threshold, integrated_signals[i], 0)
        thresholded_signals.append(thresholded_signal)

    return  diff_signals,filtered_signals, ecg_fields, ecg_signals



def preprocess_ecg_signal_GUI(signal, fields):
    # Preprocessing
    ecg_corrected = []
    # Define bandpass filter parameters
    low_cut = 1.0
    high_cut = 40.0
    order = 2

    fs = fields['fs']
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal[:, 0])

    # Apply a high-pass filter to remove baseline wander and DC drift
    b, a = butter(2, 0.5 / (1000 / 2), 'highpass')
    ecg_filtered = filtfilt(b, a, filtered)
    ecg_corrected.append(ecg_filtered)

    # Estimate the isoelectric line (baseline) using a moving average filter
    window_size = int(1000 * 0.15)  # 200 ms window size
    baseline = savgol_filter(ecg_corrected[0], window_size, 1)

    # Subtract the estimated baseline from the filtered ECG signal
    filtered_signal = ecg_corrected[0] - baseline

    # Differentiation
    diff_signal = np.diff(filtered_signal)

    # Squaring
    squared_signal = diff_signal ** 2

    # Moving-window integration
    window_size = int(0.15 * fs)  # 150 ms window
    integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

    # Thresholding
    threshold = 0.6  # This value should be adjusted based on the data
    thresholded_signal = np.where(integrated_signal > threshold , integrated_signal, 0)

    return filtered_signal
