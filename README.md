# speech-class
Code used in the research of English dialect classification.
# Directory of Files
### Data Wrangling
<p> These files have the goal of taking .wav audio files, generating force-aligned phoneme transcriptions, and outputting sound metrics into a .csv file. This data
can then be analyzed using a series of Python classes and methods. </p>

- *note* - before the process begins labeled audio files must be put into the montreal-input and montreal-output directories <br>
- **wav_conversion.py** - a Python script with a single function `convert_mp3_to_wav()` used to convert a directory of .mp3 files into .wav files (only necessary if this is not already the case) <br>
- **textgrid_creation.py** - a Python script with a single function `create_textgrids()` used to generate a series of textgrids in the specified directory that have the specified text as a single interval <br>
- **custom_mfa_align.py** - a Python script with a single function `align()` that runs a series of Montreal Forced Aligner terminal operations to align the textgrids in a specified directory <br> 
- **formant-extraction.py** - a Praat script which extracts formant measurements and other sound metrics from each phoneme in the textgrids of the specified directory, outputting results 
into a spreadsheet named formants.csv <br>
  
### speech_modeling.py
<p> A Python script with two classes. </p>

- **h_input**
  - is initialized with the arguments `raw_data` which should be a pandas dataframe produced from the output of formant_extraction.py 
and `id_vars` which are non-value features of the output produced by formant_extraction.py <br>
  - the instance variable `self.input_df` represents the processed dataframe <br>
  - `process` method takes arguments `location_specificity="country"` and `vowels_only=True` <br>
  - `select_features` method takes the argument `selected_features=['F1']` and drops other features from `self.input_df` <br>
  - `normalize` method takes the argument `method='z'` <br>
  - if `method=='z'` then features will be normalized using standardization, if `method=='minmax'`then features will be normalized using min-max normalization <br>
  - `select_places` method takes the argument `places=["uk", "usa"]` and drops samples that aren't labeled with these locations <br>
  - `revert` method removes previous processing and brings the input dataframe back to its original state <br>
  - `output_input_df` method takes the argument `filename="input_df.csv"` <br>
- **h_model**
  - takes the dataframe from the processed `self.input_df` to be used in the implementation of different statistical models <br>
  - is initialized with the arguments `data` which should be a copy of `self.input_df`, `model_features` which should be a list of features in this dataframe to be modeled, 
`y_feature` which should be the feature to be predicted (likely location), and `y_main` which should be the particular value of `y_feature` which will be separated from the 
rest (these models perform binary classification, prediction whether a sample is labeled with `y_main` or any other label) <br>
  - the `fit` method takes arguments `model_type="rforest"`, `cv_method="LOO"`, `test_size=0.30` if the `"train-test"` cross validation method is being used, 
and `var_imp_type="mdi"` which specifies the type of variable importance metric to be calculated

### Additional Files and Directories<br>
- **gmu_scraping.py** - this file scrapes from the GMU Archive all of the English language files and their demographic information as .wav files as seen in the specified directory <br>
- **LICENSE** - basic MIT License <br>
- **README** - this file <br>
- **gmu-modeling.ipynb** - jupyter notebook used for modeling the GMU Archive data <br>
- **data** - all of the audio tracks, aligned textgrids, and output .csv files from all of the English language samples in the two example datasets <br>

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
- adapted from [this](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) tutorial by Eleanor Chodroff
