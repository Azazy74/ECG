import matplotlib.pyplot as plt


def plot_ecg_signals(ecg_signals):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Plot each ECG signal on a separate subplot
    for i in range(4):
        row = i // 2
        col = i % 2
        axs[row, col].plot(ecg_signals[i])
        axs[row, col].set_xlabel('Sample number')
        axs[row, col].set_ylabel('Signal amplitude')
        axs[row, col].set_title(f'Signal {i + 1}')

    # Adjust subplot spacing and display the plot
    plt.tight_layout()
    plt.show()


def plot_ecg_signals_duration(ecg_signals, ecg_fields, start_sec, end_sec):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    for i in range(4):
        fs = ecg_fields[i]['fs']
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs)
        row = i // 2
        col = i % 2

        axs[row, col].plot(ecg_signals[i][start_sample:end_sample])
        axs[row, col].set_xlabel('Sample number')
        axs[row, col].set_ylabel('Signal amplitude')
        axs[row, col].set_title(f'Signal {i + 1}')
        axs[row, col].set_xlim([0, end_sample - start_sample])

    plt.tight_layout()
    plt.show()



def plot_filtered_signals(filtered_signals):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    for i in range(4):
        row = i // 2
        col = i % 2
        axs[row, col].plot(filtered_signals[i])
        axs[row, col].set_xlabel('Sample number')
        axs[row, col].set_ylabel('Signal amplitude')
        axs[row, col].set_title(f'Signal {i + 1}')

    plt.tight_layout()
    plt.show()


def plot_filtered_signals_duration(filtered_signals, ecg_fields, start_sec=0, end_sec=2):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    for i in range(4):
        fs = ecg_fields[i]['fs']
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs)
        row = i // 2
        col = i % 2

        axs[row, col].plot(filtered_signals[i][start_sample:end_sample])
        axs[row, col].set_xlabel('Sample number')
        axs[row, col].set_ylabel('Signal amplitude')
        axs[row, col].set_title(f'Signal {i + 1}')
        axs[row, col].set_xlim([0, end_sample - start_sample])

    plt.tight_layout()
    plt.show()


def plot_fiducial_points(filtered_signals, R_indices, Q_indices, S_indices, T_indices, P_indices, P_onset, P_offset, T_onset, T_offset,QRS_onset, QRS_offset):
    for i in range(4):
        plt.plot(filtered_signals[i])
        plt.scatter(R_indices[i], filtered_signals[i][R_indices[i]], c='red')
        plt.scatter(Q_indices[i], filtered_signals[i][Q_indices[i]], c='green')
        plt.scatter(S_indices[i], filtered_signals[i][S_indices[i]], c='blue')
        plt.scatter(T_indices[i], filtered_signals[i][T_indices[i]], c='cyan')
        plt.scatter(P_indices[i], filtered_signals[i][P_indices[i]], c='magenta')
        plt.scatter(P_onset[i], filtered_signals[i][P_onset[i]], c='yellow')
        plt.scatter(P_offset[i], filtered_signals[i][P_offset[i]], c='orange')
        plt.scatter(T_onset[i], filtered_signals[i][T_onset[i]], c='purple')
        plt.scatter(T_offset[i], filtered_signals[i][T_offset[i]], c='brown')
        plt.scatter(QRS_onset[i], filtered_signals[i][QRS_onset[i]], c='pink')
        plt.scatter(QRS_offset[i], filtered_signals[i][QRS_offset[i]], c='gray')

        plt.xlabel('Sample number')
        plt.ylabel('Amplitude')
        plt.xlim(1000, 2200)
        plt.title('ECG Signal with Fiducial Points')
        plt.show()
