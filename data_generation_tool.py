import os
import random
import threading
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/audio_files", exist_ok=True)
output_csv = "outputs/recorded_samples.csv"
output_csv_file = open(output_csv, "a")
if os.path.getsize(output_csv) == 0:
    output_csv_file.write("audio_path,text\n")
    print("CSV file created.")
text_dataset_samples = []
with open("text_samples.txt", "r", encoding='utf-8') as f:
    text_dataset_samples = f.read()
    text_dataset_samples = text_dataset_samples.splitlines()

is_recording = False
def record_audio(output_file_path):
    global is_recording
    fs = 16000  # Sample rate
    block_size = 480  # Block size for audio stream
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', blocksize=block_size) as stream:
        audio_data = []
        while is_recording:
            block, overflowed = stream.read(block_size)
            if overflowed:
                print("Warning: input overflow")
            audio_data.append(block.copy())

    audio_data = np.concatenate(audio_data)
    # save audio_data to file
    sf.write(output_file_path, audio_data, fs)


if __name__ == "__main__":
    while True:
        text = random.choice(text_dataset_samples)
        print("Generating text sample...")
        print("Sample text: ", text)
        key = input("Enter c to start recording, q to quit...")
        if key == 'c':
            is_recording = True
            output_file_path = f"{output_dir}/audio_files/sample_{int(time.time())}.wav"
            recording_thread = threading.Thread(target=record_audio, args=(output_file_path,))
            recording_thread.start()
            key = input("Enter any key to stop recording...")
            print("Stopping recording...")
            is_recording = False
            recording_thread.join()
            print("Recording stopped.")
            print(f"Audio saved to {output_file_path}")
            output_csv_file.write(f"{output_file_path},\"{text.strip()}\"\n")
            output_csv_file.flush()
        elif key == 'q':
            print("Exiting...")
            break
                