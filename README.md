# speech-class
Code used in the research of English dialect classification.
# Directory of Files
### Main Process
<p> These files have the goal of taking .wav audio files, generating force-aligned phoneme transcriptions, and outputting sound metrics into a .csv file. This data
can then be analyzed using a series of Python classes and methods. </p>

- *note* - before the process begins labeled audio files must be put into the montreal-input and montreal-output directories <br>
- **wav_conversion.py** - a Python script with a single function `convert_mp3_to_wav()` used to convert a directory of .mp3 files into .wav files (only necessary if this is not already the case) <br>
- **textgrid_creation.py** - a Python script with a single function `create_textgrids()` used to generate a series of textgrids in the specified directory that have the specified text as a single interval <br>
- **custom_mfa_align.py** - a Python script with a single function `align()` that runs a series of Montreal Forced Aligner terminal operations to align the textgrids in a specified directory <br> 
- **formant-extraction** - a Praat script which extracts formant measurements and other sound metrics from each phoneme in the textgrids of the specified directory, outputting results //
into a spreadsheet named formants.csv <br>
- **speech_modeling.py** - a Python script with two classes <br>
  - **h_input**
  - `h_input` - takes a dataframe of formants.csv and allows for various modifications of the raw data <br>
  - the instance variable `self.input_df` represents the processed dataframe <br>
  - `process` method takes a single argument `drop_cols=True` <br>
  - if `drop_cols==True` then features with majority zero values will be dropped from the input dataframe <br>
  - `normalize` method takes a single argument `method='z'` <br>
  - if `method=='z'` then features will be normalized using standardization, if `method=='minmax'`then features will be normalized using min-max normalization <br>
  - `revert` method removes previous processing and brings the input dataframe back to its original state <br>
  - **h_model**
  - `h_model` <br>

### Additional Files <br>
- **gmu_scraping.py** - this file scrapes all of the English language files and their demographic information as .wav files as seen in the specified directory <br>
- **LICENSE** - basic MIT License <br>
- **README** - this file <br>
- **testing.ipynb** - jupyter notebook used for testing python code <br>
- **formants.csv** - this is the output file of the praat script which <br>

### Praat Scripts
<p> These files correspond to Python files but with less functionality and running in Praat.</p>

- **wav-conversion.praat** <br>
- **textgrid-creation.praat** <br>
-  **formant-extraction-praat** <br>

### Notes
- Text alignment can be done without the custom_mfa_align.py file using the Montreal Forced Aligner directly in the terminal. <br>
- The following lines of code download the models necessary for forced alignment. Other models may be used. <br>
`mfa model download acoustic english_us_arpa` <br>
`mfa model download dictionary english_us_arpa` <br>
`mfa align --clean CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY` implements the forced aligner <br>
`mfa align --clean /home/user/montreal-input english_us_arpa english_us_arpa /home/user/montreal-output` (example) <br>
