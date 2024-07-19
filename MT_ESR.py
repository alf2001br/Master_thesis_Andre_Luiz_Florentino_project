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

import tkinter     as tk

from IPython.display import display
from queue           import Queue
from threading       import Thread


# Globals
current_path   = os.getcwd()

os_name = os.name

if os_name == 'posix':
    CHUNK = 4096
else:
    CHUNK = 1024

print(f'\nChunk size {CHUNK}\n')

FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = 44100 # Higher sampling rate for recording live audio
RATE_ESR       = 22050 # Lower sampling rate to match the predicting models
RECORD_SECONDS = 1
SAMPLE_SIZE    = 2
frames         = []

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


# Path were the trained models and arrays are stored
path_modelsVal = os.path.join(current_path, "_ESR", "Saved_models_fold_1_validation")
path_arrays    = os.path.join(current_path, "_ESR", "Arrays")

# Load the ESR algorythm
from MT_ESR_evaluation_tflite import ESR_evaluation_tflite

# Cache folder to store the temporary recorded audio
cache_audio = os.path.join(current_path, '_cache_record_audio')

# Check if the folder exists, if not, create it
if not os.path.exists(cache_audio):
    os.makedirs(cache_audio)

"==================================== HELPER FUNCTIONS ======================================="

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
            data = stream.read(chunk)
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


# Start the ESR (live prediction)
def ESR():
    try:
        while not messages.empty():
            frames = recordings.get()

            # Save the audio file as .WAV for loading into Librosa
            wf = wave.open(os.path.join(cache_audio, "output.wav"), "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            rawdata, _ = librosa.load(os.path.join(cache_audio, 'output.wav'), sr=RATE_ESR, mono=True)

            print("================================================")
            print(f'Sound duration..: {librosa.get_duration(y=rawdata):2f}\n')
            ESR_evaluation_tflite([rawdata], 'CNN2D', path_modelsVal, path_arrays)

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
