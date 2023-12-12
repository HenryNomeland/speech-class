# read all mp3 files from input directory and convert to wav before deleting all mp3 files

inputPath$ =  "/home/henrynomeland/Documents/senior-thesis/speech-class/mp3-input" #input directory

outputPath$ = "/home/henrynomeland/Documents/senior-thesis/speech-class/montreal-input" #output directory

inPath$ = inputPath$ + "/"
mp3Path$ = inPath$ + "*.mp3"
soundList = Create Strings as file list: "soundList", mp3Path$
numSounds = Get number of strings

for soundNum from 1 to numSounds

	selectObject: soundList

	#get sound file and read in sound file
	soundName$ = Get string: soundNum

	#convert and save to mp3
	mp3 = Read from file: inPath$ + soundName$
	newName$ = soundName$ - ".mp3"

	Write to WAV file... 'outputPath$'/'newName$'.wav

endfor
