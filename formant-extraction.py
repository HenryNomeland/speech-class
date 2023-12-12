# goes through every .wav and .textgrid file in a directory and extracts formant measurements to a .csv
# is the python version of the praat script with the same name but with better functionality

import os
import parselmouth
from parselmouth.praat import call
import pandas as pd
import tgt

def extract(folder, output_folder, filename):
    filepath = os.path.join(output_folder, filename)
    df = pd.DataFrame(data=None, columns=["id","gender","age","location",
                                          "midpoint_time","phoneme","word",
                                          "duration","F1","F2","F3","F1_onglide",
                                          "F2_onglide","F1_offglide","F2_offglide"])
    for file in os.listdir(folder):
        if file.endswith(".TextGrid"):
            filelist = file.split("-")
            wav_path = os.path.join(folder, file[:-8] + "wav")
            sound = parselmouth.Sound(wav_path)
            tg_path = os.path.join(folder, file)
            textgrid = tgt.read_textgrid(tg_path)
            formants = sound.to_formant_burg()
    
            valueHz = 5000
            if filelist[1] == "f":
                valueHz = 5500
            formants = call(sound, "To Formant (burg)...", 0, 5, valueHz, 0.025, 50)

            phone_intervals = textgrid.get_tier_by_name("phones").intervals
            word_tier = textgrid.get_tier_by_name("words")
            for interval in phone_intervals:
                start_time = interval.start_time
                end_time = interval.end_time
                phoneme = interval.text
                id = filelist[0]
                gender = filelist[1]
                age = filelist[2]
                location = filelist[3]
                duration = end_time - start_time
                midpoint = start_time + duration*0.5
                onglide = start_time + duration*0.2
                offglide = start_time + duration*0.8
    
                word = word_tier.get_annotations_by_time(midpoint)[0].text
                f1 = call(formants, "Get value at time...", 1, midpoint, "Hertz", "Linear")
                f2 = call(formants, "Get value at time...", 2, midpoint, "Hertz", "Linear")
                f3 = call(formants, "Get value at time...", 3, midpoint, "Hertz", "Linear")
                f1on = call(formants, "Get value at time...", 1, onglide, "Hertz", "Linear")
                f2on = call(formants, "Get value at time...", 2, onglide, "Hertz", "Linear")
                f1off = call(formants, "Get value at time...", 1, offglide, "Hertz", "Linear")
                f2off = call(formants, "Get value at time...", 2, offglide, "Hertz", "Linear")
    
                df.loc[len(df.index)] = [id, gender, age, location, midpoint, phoneme, word, duration, 
                                         f1, f2, f3, f1on, f2on, f1off, f2off] 

        df.to_csv(filepath, index=False)
            
                
if __name__ == "__main__":
    folder = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-output"
    output_folder = "/home/henrynomeland/Documents/senior-thesis/speech-class" 
    filename = "testing.csv"

    extract(folder, output_folder, filename)

    print("Extraction Successful")