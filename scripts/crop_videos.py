#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:32:02 2021

@author: bendeskylab
"""
import pandas as pd
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

os.chdir('/Users/claireeverett/Desktop/Process_input/videos/')

import sys
exp_name = sys.argv[1]
print(exp_name)

# import csv with the crop coord
test = pd.read_csv('/Users/claireeverett/Desktop/Process_input/videos/test_coordinates.csv')

# path to the folder of trials
path = '/Users/claireeverett/Desktop/Process_input/videos/'

# loop through the csv sheet of coord as well as list of trial folders - must make sure order is the same 
counter = 0
for dir in next(os.walk('.'))[1]:
    print(dir)
    if dir.startswith(exp_name):
        for file in os.listdir(dir):
            if file.endswith(".h264"):
                os.system('ffmpeg -framerate 40 -i {0} -c copy {0}.mp4'.format(os.path.join(path, dir,file))) 

for dir in sorted(next(os.walk('.'))[1]):
    if dir.startswith(exp_name):
        name_list = []
        for file in os.listdir(dir):
            if file.endswith(".mp4"):
                clip = (VideoFileClip(os.path.join(path, dir,file)).crop(x1=test['x'][counter],y1=test['y'][counter], x2=test['x'][counter]+500, y2=test['y'][counter]+500))
                clip.write_videofile(str(counter) + file[:10] + file[-15:-9] + "crop.mp4") # indexing into the file names will be different from trial to trial - find right index to name properly
                name = str(counter) + file[:10] + file[-15:-9] + "crop.mp4"
                name_list.append('file ' + name)
        name_list.sort()
        with open('output.txt', 'w') as filehandle:
            for listitem in name_list:
                filehandle.write('%s\n' % listitem)
        os.system('ffmpeg -f concat -i output.txt -vcodec copy -acodec copy {0}.mp4'.format(dir))
        os.system('rm {0}*'.format(counter))
        counter = counter + 1
            
    