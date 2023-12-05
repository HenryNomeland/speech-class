# testing out scraping
import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import urllib

class getFiles:
    
    def __init__(self, destination="/home/henrynomeland/Documents/senior-thesis/corpus-testing/montreal-input",
                       alt_destination="/home/henrynomeland/Documents/senior-thesis/corpus-testing/montreal-output"):
        self.destination = destination
        self.alt = alt_destination
       
    def get_audio(self):
        
        base = "https://accent.gmu.edu/"
        page = requests.get("https://accent.gmu.edu/browse_language.php?function=find&language=english")
        soup = BeautifulSoup(page.text, 'html.parser')
        
        for tag in soup.findAll('p'):
            a = tag.a
            if a != None:
                newlink = base + a.attrs['href']
                newpage = requests.get(newlink)
                newsoup = BeautifulSoup(newpage.text, 'html.parser')
                
                list_items = newsoup.find("ul", {"class": "bio"}).find_all('li')
                location_list = list_items[0].contents[1].split(',')
                location1 = ''.join(filter(lambda x: x.isalpha(), location_list[0].lower()))
                location2 = ''.join(filter(lambda x: x.isalpha(), location_list[1].lower()))
                if len(location_list) > 2:
                    location3 = ''.join(filter(lambda x: x.isalpha(), location_list[2].lower()))
                tempage = list_items[3].contents[1].split(',')[0]
                age = tempage[1:3]
                if int(age) < 18:
                    continue
                
                gender = list_items[3].contents[1].split(',')[1]
                mf = gender[1]
                
                aud = newsoup.find('audio').source
                subname = aud.attrs['src']
                name = base[:-1] + subname
                (filename, headers) = urllib.request.urlretrieve(name)
                sound = AudioSegment.from_mp3(filename)
                
                if len(location_list) > 2:
                    newname = f"{subname[12:-4]}-{gender[1]}-{age}-{location1}_{location2}_{location3}"
                else:
                    newname = f"{subname[12:-4]}-{gender[1]}-{age}-{location1}_{location2}"
                print(newname)
                if not os.path.exists(self.destination + '{}.wav'.format(newname)) | \
                       os.path.exists(self.alt + '{}.wav'.format(newname)):
                    sound.export(self.destination + "{}.wav".format(newname), format="wav")
                    sound.export(self.alt + "{}.wav".format(newname), format="wav")

        print("Success")

if __name__ == "__main__":
    getFiles().get_audio()
    print("Operation Successful")
        