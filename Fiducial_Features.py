import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import preprocessing_data



def fiducial_features_extraction(filtered_signals):
    fs = 1000

    def pan_and_tompkins(signals):
        convolved_Signals = []
        for i in signals:
            dx = np.diff(i)
            dx = np.square(dx)
            dx = np.convolve(dx, np.ones(200), mode='same')
            convolved_Signals.append(dx)
            plt.plot(dx[0:3000])
            plt.show()
        return convolved_Signals

    convolved_Signals = pan_and_tompkins(filtered_signals)

    # Fiducial Feature Extraction
    R_indices = []
    for i in range(4):
        peaks, _ = find_peaks(filtered_signals[i], distance=550)
        R_indices.append(peaks)

    S_indices = []
    for i in range(4):
        s_idx = []
        for r_peak in R_indices[i]:
            window_size = int(0.04 * fs)
            local_min_index = np.argmin(filtered_signals[i][r_peak:r_peak+window_size])
            s_idx.append(r_peak + local_min_index)
        S_indices.append(s_idx)

    Q_indices = []
    for i in range(4):
        q_idx = []
        for r_peak in R_indices[i]:
            window_size = int(0.1 * fs)
            start = r_peak - window_size
            if start < 0:
                start = 0
            local_min_index = np.argmin(filtered_signals[i][start:r_peak -1])
            q_idx.append(start + local_min_index)
        Q_indices.append(q_idx)

    T_indices = []
    for i in range(4):
        t_idx = []
        for r_peak in R_indices[i]:
            window_size = int(0.4 * fs)
            local_max_index = np.argmax(filtered_signals[i][r_peak + 40:r_peak+ 40 + window_size])
            t_idx.append(r_peak + 40 + local_max_index)
        T_indices.append(t_idx)

    P_indices = []
    for i in range(4):
        p_idx = []
        for r_peak in R_indices[i]:
            window_size = int(0.2 * fs)
            start = r_peak - window_size - 40
            if start < 0:
                start = 0
            local_max_index = np.argmax(filtered_signals[i][start:r_peak -40])
            p_idx.append(start + local_max_index)
        P_indices.append(p_idx)


    def calculate_waves_onset_and_offset(signals, wave_indices, search_space):
        onsets = []
        offsets = []
        for i in range(len(signals)):
            onset = []
            offset = []
            for peak in wave_indices[i]:
                xx = (peak, filtered_signals[i][peak]) # peak
                y_left = (peak - search_space, filtered_signals[i][peak - search_space])
                s = min(peak+search_space, len(filtered_signals[i]) -1 )
                y_right = (s, filtered_signals[i][s])
                a_vector_left = (y_left[0] - xx[0], y_left[1] - xx[1])
                a_vector_right = (y_right[0] - xx[0], y_right[1] - xx[1])

                maximum_left = maximum_right =  max_idx_left = max_idx_right = -1
                for point in range(search_space):
                    care2 = min(peak+point, len(filtered_signals[i]) - 1)
                    m_left = (peak-point, filtered_signals[i][peak-point])
                    m_right = (care2, filtered_signals[i][care2])
                    c_vector_left = (m_left[0] - xx[0], m_left[1] - xx[1])
                    c_vector_right = (m_right[0] - xx[0], m_right[1] - xx[1])
                    cross_left = np.cross(a_vector_left, c_vector_left)
                    cross_right = np.cross(a_vector_right, c_vector_right)
                    sigma_left = np.abs(cross_left) / np.abs(np.linalg.norm(a_vector_left))
                    sigma_right = np.abs(cross_right) / np.abs(np.linalg.norm(a_vector_right))

                    if sigma_left > maximum_left:
                        maximum_left = sigma_left
                        max_idx_left = peak - point

                    if sigma_right > maximum_right:
                        maximum_right = sigma_right
                        max_idx_right = care2

                onset.append(max_idx_left)
                offset.append(max_idx_right)

            onsets.append(onset)
            offsets.append(offset)

        return onsets, offsets

    P_onset, P_offset = calculate_waves_onset_and_offset(filtered_signals, P_indices, 80)
    T_onset, T_offset = calculate_waves_onset_and_offset(filtered_signals, T_indices, 180)
    QRS_onset, QRS_offset = calculate_waves_onset_and_offset(convolved_Signals,R_indices,120)
    features = []
    classes = []

    for i in range(4):
        for j in range(len(P_indices[i])):
            QT_duration = (T_indices[i][j]  - Q_indices[i][j]) / fs
            PQ_duration = ((Q_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            PR_duration = ((R_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            PS_duration = ((S_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            PT_duration = ((T_indices[i][j] - P_indices[i][j]) / fs) / QT_duration
            QS_duration = ((S_indices[i][j] - Q_indices[i][j]) / fs) / QT_duration
            QR_duration = ((R_indices[i][j] - Q_indices[i][j]) / fs) / QT_duration
            RS_duration = ((S_indices[i][j] - R_indices[i][j]) / fs) / QT_duration
            RT_duration = ((T_indices[i][j] - R_indices[i][j]) / fs) / QT_duration
            RP_freq = (filtered_signals[i][R_indices[i][j]] - filtered_signals[i][P_indices[i][j]])
            RT_freq = (filtered_signals[i][R_indices[i][j]] - filtered_signals[i][T_indices[i][j]])
            TP_freq = (filtered_signals[i][T_indices[i][j]] - filtered_signals[i][P_indices[i][j]])
            heart_beat_features = [QT_duration, PQ_duration, PR_duration, PS_duration,
                                   PT_duration, QS_duration, QR_duration, RS_duration, RT_duration,
                                   RP_freq, RT_freq, TP_freq ]
            features.append(heart_beat_features)
            classes.append(i)

    return features, classes , P_onset, P_offset ,T_onset, T_offset ,P_indices,R_indices,Q_indices,S_indices,T_indices,QRS_onset, QRS_offset


