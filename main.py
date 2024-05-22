import wfdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import preprocessing_data
import Fiducial_Features
import Visualization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import classfication

#Preprocessing Data
data_path = 'Data'
diff_signals,filtered_signals,ecg_fields,ecg_signals = preprocessing_data.preprocess_ecg_data(data_path)


#Visualization
#----------------------

#Before preprocessing
Visualization.plot_ecg_signals(ecg_signals)
Visualization.plot_ecg_signals_duration(ecg_signals, ecg_fields, start_sec=0, end_sec=2)

#After preprocessing
Visualization.plot_filtered_signals(filtered_signals)
Visualization.plot_filtered_signals_duration(filtered_signals, ecg_fields, start_sec=5, end_sec=7)

# Call the function with filtered_signals as argument
features,classes,P_onset,P_offset,T_onset,T_offset,P_indices,R_indices,Q_indices,S_indices,T_indices,QRS_onset,QRS_offset = Fiducial_Features.fiducial_features_extraction(filtered_signals)


#Plot all indices
Visualization.plot_fiducial_points(filtered_signals, R_indices, Q_indices, S_indices, T_indices, P_indices, P_onset, P_offset, T_onset, T_offset,QRS_onset, QRS_offset)

#X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.2, shuffle=True)

# Load Models
#model1 = joblib.load('Models\SVM.joblib')
#model2 = joblib.load('Models\LDA.joblib')
#model3 = joblib.load('Models\LogisticRegression.joblib')

# Perdict Labels
#y_pred1 = model1.predict(X_test)
#y_pred2 = model2.predict(X_test)
#y_pred3 = model3.predict(X_test)

# Calculate Accuracy
#accuracy1 = accuracy_score(y_test, y_pred1)
#accuracy2 = accuracy_score(y_test, y_pred2)
#accuracy3 = accuracy_score(y_test, y_pred3)
#print(f'Accuracy of SVM: {accuracy1 * 100:.2f}%')
#print(f'Accuracy of LDA: {accuracy2 * 100:.2f}%')
#print(f'Accuracy of LogisticRegression: {accuracy3 * 100:.2f}%')

#Classification Train
classfication.train_and_evaluate_models(features, classes)

def compute_similarity(signal1, signal2):
    # Convert the two matrices into vectors
    signal1 = np.squeeze(signal1)
    signal2 = np.squeeze(signal2)

    #Calculate the cosine similarity coefficient
    dot_product = np.dot(signal1, signal2)
    norm_signal1 = np.linalg.norm(signal1)
    norm_signal2 = np.linalg.norm(signal2)
    similarity = dot_product / (norm_signal1 * norm_signal2)

    return similarity
def load_and_match_signal():
    file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])

    if file_path:
        signal, fields = wfdb.rdsamp(file_path.split('.')[0], channels=[0])
        filtered_signal = preprocessing_data.preprocess_ecg_signal_GUI(signal, fields)

        # Compute similarity with each signal in your dataset
        similarities = []
        similarities = [compute_similarity(signal, filtered_signal) for signal in filtered_signals]
        # Find the index of the most similar signal
        max_sim_index = similarities.index(max(similarities))

        # Display the most similar signal inside the GUI
        if max(similarities) > 0.5:
            match_label.config(text="Identified signal.", fg="green")
            plt.figure()
            plt.plot(ecg_signals[max_sim_index][:, 0])
            plt.xlabel('Sample number')
            plt.ylabel('Signal amplitude')
            plt.title(f'Matched Signal {max_sim_index + 1}')
            canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            match_label.config(text="No matching signals found.", fg="red")


# Create the GUI window
root = tk.Tk()
root.title("ECG Signal Matcher")

# Add a button to load and match signals
load_button = tk.Button(root, text="Load and Match Signal", command=load_and_match_signal)
load_button.pack()

# Add a label to display match status
match_label = tk.Label(root, text="")
match_label.pack()

# Add a frame to hold the plot
plot_frame = tk.Frame(root)
plot_frame.pack()

# Run the GUI main loop
root.mainloop()
