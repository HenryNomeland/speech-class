# read all wav files from a directory and generate a simple textgrid for each
# is the python version of the praat script with the same name but with better functionality

import os
import parselmouth
from parselmouth.praat import call

def create_textgrids(label, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            wav_path = os.path.join(folder, filename)
            destination = os.path.join(folder, filename[:-4] + ".TextGrid")
            sound = parselmouth.Sound(wav_path)
            textgrid = call(sound, "To TextGrid...", "segment", "None")
            call(textgrid, "Set interval text", 1, 1, label)
            call(textgrid, "Save as text file...", destination)

    print("TextGrid Creation Successful")

if __name__ == "__main__":
    folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-input"
    label = "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."

    create_textgrids(label, folder)