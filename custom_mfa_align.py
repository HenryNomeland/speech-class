# read all textgrids and wav files in a folder and align them using mfa

import os
import subprocess

def align(in_folder, out_folder, acoustic_model="english_us_arpa", dictionary_model="english_us_arpa"):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    subprocess.call(f"mfa model download acoustic {acoustic_model}", 
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    subprocess.call(f"mfa model download dictionary {dictionary_model}", 
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    subprocess.call(f"mfa align --clean {in_folder} {acoustic_model} {dictionary_model} {out_folder}", 
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("Alignment Successful")

if __name__ == "__main__":
    in_folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-input"
    out_folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-output"
    acoustic_model = "english_us_arpa"
    dictionary_model = "english_us_arpa"

    align(in_folder, out_folder, acoustic_model, dictionary_model)