# read all mp3 files from input directory and convert to wav before deleting all mp3 files
# is the python version of the praat script with the same name but with better functionality

from pydub import AudioSegment
import os

def convert_mp3_to_wav(in_folder, out_folder, extra_folder="NA"):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for filename in os.listdir(in_folder):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(in_folder, filename)
            audio = AudioSegment.from_mp3(mp3_path)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(out_folder, wav_filename)
            audio.export(wav_path, format="wav")
            if extra_folder != "NA":
                if not os.path.exists(extra_folder):
                    os.makedirs(extra_folder)
                extra_path = os.path.join(extra_folder, wav_filename)
                audio.export(extra_path, format="wav")
                
if __name__ == "__main__":
    in_folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/mp3-input"
    out_folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-input"
    extra_folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-output"

    convert_mp3_to_wav(in_folder, out_folder, extra_folder)

    print("Conversion Successful")