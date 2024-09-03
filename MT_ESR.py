"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

# REMARKS

# This script is based on the notebook 15_ESR.ipynb, and it was created to run in both notebook and Raspberry Pi.


# Main libraries
import pyaudio
import os
import librosa
import wave
import csv
import re
import datetime

import numpy   as np
import tkinter as tk

from IPython.display import display
from queue           import Queue
from threading       import Thread


# Globals
current_path   = os.getcwd()

os_name = os.name

if os_name == 'posix':
    CHUNK = 1024*8 # Higher chunk to avoid overflow in the Raspberry Pi
else:
    CHUNK = 1024

print(f'\nChunk size {CHUNK}\n')

FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = 44100 # Higher sampling rate for recording live audio
RATE_ESR       = 22050 # Lower sampling rate to match the predicting models
RECORD_SECONDS = 1
frames         = []

predictions_array   = []
totalPredTime_array = []

# Retrieve the cateforical classes
path_arrays = os.path.join(current_path, "_ESR", "Arrays")
nom_classes = []
with open(os.path.join(path_arrays, 'nom_classes.csv'), 'r') as file:
    for line in file:
        nom_classes.append(line.strip())

# Path were the trained models
path_modelsVal = os.path.join(current_path, "_ESR", "Saved_models_fold_1_validation")

# Path were the live predictions are saved
path_live_pred = os.path.join(current_path, "_ESR", "Live_predictions")

# Check if the folder exists, if not, create it
if not os.path.exists(path_live_pred):
    os.makedirs(path_live_pred)

# Find audio device index
devices = []
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    devices.append(p.get_device_info_by_index(i))
p.terminate()

print(f'\nDevice used in this algorithm to record live audio:\n')
for item in devices[1].items():
    print(item)
print()
print("=============================================================================================\n")




# Load the ESR algorythm
from MT_ESR_evaluation_tflite import ESR_evaluation_tflite

# Cache folder to store the temporary recorded audio
cache_audio = os.path.join(current_path, '_cache_record_audio')

# Check if the folder exists, if not, create it
if not os.path.exists(cache_audio):
    os.makedirs(cache_audio)

"==================================== HELPER FUNCTIONS ======================================="

# Function to check the counter of the files _live_audio_predictions.csv and _live_audio_totalPredTime.csv
def get_next_counter(path, prefix, extension):
    files = os.listdir(path)
    pattern = re.compile(rf"{prefix}_(\d+)\.{extension}")
    counters = [int(pattern.search(f).group(1)) for f in files if pattern.search(f)]
    return max(counters) + 1 if counters else 0


# Function to record the audio from a specific microphone
def record_microphone(chunk=CHUNK):
    try:
        p = pyaudio.PyAudio()

        stream = p.open(format             = FORMAT,
                        channels           = CHANNELS,
                        rate               = RATE,
                        input              = True,
                        input_device_index = devices[1]['index'],
                        frames_per_buffer  = CHUNK)

        global frames

        while not messages.empty():
            data = stream.read(chunk)  # exception_on_overflow=False allow decreasing the chunk size
            frames.append(data)
            if len(frames) >= (RATE * RECORD_SECONDS) / chunk:
                recordings.put(frames.copy())
                frames = []

        stream.stop_stream()
        stream.close()
        p.terminate()

    except Exception as e:
        print(f"Error during recording: {e}")


# Start the thread recording
def start_recording():
    messages.put(True)

    display("Starting...")

    # Here starts the live audio recording
    record = Thread(target=record_microphone)
    record.start()

    # Here starts the ESR algorithm
    esr = Thread(target=ESR)
    esr.start()


# Stop the thread recording
def stop_recording():
    if not messages.empty():

        messages.get()
        display("Stopped.")

    # Access and print global arrays after stopping recording
    global predictions_array, totalPredTime_array
    print("\nPredictions Array:", predictions_array)
    print("Total Prediction Time Array:", totalPredTime_array)

    # Determine the next file counter
    pred_counter = get_next_counter(path_live_pred, '_live_audio_predictions', 'csv')
    time_counter = get_next_counter(path_live_pred, '_live_audio_totalPredTime', 'csv')

    # Ensure both counters are in sync
    counter = max(pred_counter, time_counter)

    # Write the result to a file (array format)
    np.array(predictions_array).tofile(os.path.join(path_live_pred, f'_live_audio_predictions_{counter}.csv'), sep=',')
    np.array(totalPredTime_array).tofile(os.path.join(path_live_pred, f'_live_audio_totalPredTime_{counter}.csv'), sep=',')

    # Write the result to a file (line by line format)
    # np.savetxt(os.path.join(path_live_pred, '_live_audio_predictions.csv'), np.array(predictions_array), delimiter=';', fmt='%s')
    # np.savetxt(os.path.join(path_live_pred, '_live_audio_totalPredTime.csv'), np.array(totalPredTime_array), delimiter=';')


# Start the ESR (live prediction)
def ESR():
    global predictions_array, totalPredTime_array
    try:
        count_samples = 0
        while not messages.empty():

            classifier = 'CNN2D'
            frames     = recordings.get()

            # Save the audio file as .WAV for loading into Librosa
            count_samples += 1
            print("================================================")
            print(f'\nAudio sample.........: {count_samples}')
            wav_file = str(datetime.datetime.now().strftime("%d%m%y_%H%M%S")) + '_output_' + str(count_samples) + '.wav'
            wf = wave.open(os.path.join(cache_audio, wav_file), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            rawdata, _ = librosa.load(os.path.join(cache_audio, wav_file), sr=RATE_ESR, mono=True)
            print(f'Sound duration.......: {librosa.get_duration(y=rawdata):2f}\n')
            ESR_EVAL      = ESR_evaluation_tflite([rawdata], classifier, path_modelsVal, path_arrays)

            # Return the predictions and total time for the predictions
            predictions   = np.array(ESR_EVAL.predictions)
            totalPredTime = np.array(ESR_EVAL.totalPredTime)

            # Append the results
            predictions_array.append(nom_classes[predictions[0]])
            totalPredTime_array.append(totalPredTime.tolist()[0])

    except Exception as e:
        print(f"Error during ESR: {e}")

"========================================== START ============================================"

# Initialize the main window
root = tk.Tk()
root.geometry("400x200")
root.title("ESR - Environmental Sound Recognition")

# Pass the variables for the thread
messages   = Queue()
recordings = Queue()

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(expand=True, fill='both')

# Create a nested frame to center the buttons
nested_frame = tk.Frame(button_frame)
nested_frame.pack(expand=True)

# Create buttons
record_button = tk.Button(nested_frame,
                          text    = "Record live audio",
                          command = start_recording,
                          bg      = "green",
                          fg      = "white",
                          font    = ("Helvetica", 16))

stop_button   = tk.Button(nested_frame,
                          text     = "Stop recording",
                          command = stop_recording,
                          bg      = "orange",
                          fg      = "white",
                          font    = ("Helvetica", 16))

# Place buttons in the window
record_button.pack(pady=10)
stop_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
