
# select input directory and put all .wav files into a list
# arguments
outputFileName$ = "formants.csv"
inputPath$ = "/home/henrynomeland/Documents/senior-thesis/corpus-testing/montreal-output"
outputPath$ = "/home/henrynomeland/Documents/senior-thesis/corpus-testing/" + outputFileName$

deleteFile: outputPath$
appendFileLine: "'outputPath$'", "id,gender,age,location,midpoint_time,phoneme,word,duration,F1,F2,F3,F1_onglide,F2_onglide,F1_offglide,F2_onglide"
inPath$ = inputPath$ + "/"
wavPath$ = inPath$ + "*.wav"
soundList = Create Strings as file list: "soundList", wavPath$
numSounds = Get number of strings

# loop through all .wav files and corresponding textgrids
for soundNum from 1 to numSounds
	selectObject: soundList

	# get sound file name and read in sound file
	soundName$ = Get string: soundNum

	if (fileReadable: inPath$ + soundName$)
		wav = Read from file: inPath$ + soundName$
		dot = rindex(soundName$, ".")
		dotMinus = dot - 1
		noExtension$ = left$ (soundName$, dotMinus)
	endif

	# get textgrid file name and read in textgrid
	textGridName$ = noExtension$ + ".TextGrid"
	if (fileReadable: inPath$ + textGridName$)
		tg = Read from file: inPath$ + textGridName$

		# get id, gender, age, and location
		dashOne = index(soundName$, "-")
		dashTwo = dashOne + 2
		dashThree = rindex(soundName$, "-")
		id$ = left$ (soundName$, dashOne-1 )
		gender$ = mid$ (soundName$, dashOne+1, 1)
		numAge = dashThree - dashTwo -1
		age$ = mid$ (soundName$, dashTwo+1, numAge)
		numLocation = dot - dashThree -1
		location$ = mid$(soundName$, dashThree+1, numLocation)

		# maximum number of formants = 5 Hz for m = 5000 Hz for f = 5500
		select Sound 'noExtension$'
		valueHz = 5000
		if gender$ == "f"
			valueHz = 5500
		endif
		To Formant (burg)... 0 5 valueHz 0.025 50

		# find the number of intervals
		select TextGrid 'noExtension$'
		numIntervals = Get number of intervals: 2

		# looping through the intervals
		for thisInterval from 1 to numIntervals
	
			# get the phoneme label
			select TextGrid 'noExtension$'
			thisPhoneme$ = Get label of interval: 2, thisInterval
		
			# check if the interval is a phoneme
			if thisPhoneme$ <> ""

				# find midpoint
				startTime = Get start point: 2, thisInterval
				endTime   = Get end point:  2, thisInterval
				duration = endTime - startTime
				midpoint = startTime + duration*0.5
				onglide = startTime + duration*0.2
				offglide = startTime + duration*0.8

				# get word
				wordInterval = Get interval at time: 1, midpoint
				thisWord$ = Get label of interval: 1, wordInterval

				# measure formants
        			select Formant 'noExtension$'
    				f1 = Get value at time... 1 midpoint Hertz Linear
    				f2 = Get value at time... 2 midpoint Hertz Linear
    				f3 = Get value at time... 3 midpoint Hertz Linear
				f1On = Get value at time... 1 onglide Hertz Linear
    				f2On = Get value at time... 2 onglide Hertz Linear
				f1Off = Get value at time... 1 offglide Hertz Linear
    				f2Off = Get value at time... 2 offglide Hertz Linear

				# Save to a spreadsheet
   				appendFileLine: "'outputPath$'", id$, ",",  gender$, ",",  age$, ",",  location$, ",",  midpoint, ",",  thisPhoneme$, ",", thisWord$, ",", duration, ",", 
							... f1, ",", f2, ",", f3, ",", f1On, ",", f2On, ",", f1Off, ",", f2Off

			endif
		endfor
	endif
endfor

selectObject: soundList
Remove