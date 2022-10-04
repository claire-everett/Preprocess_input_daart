#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.chdir('/Users/claireeverett/Desktop/Process_input/videos/')

path = '/Users/claireeverett/Desktop/Process_input/videos/'

for file in os.listdir():
        if file.endswith(".mp4"):
            os.system('ffmpeg -i {0} -filter_complex "[0:v]trim=start_frame=72000:end_frame=144000,setpts=PTS-STARTPTS[v0];[0:v]trim=start_frame=0:end_frame=0,setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[v]" -map "[v]" {0}_labeled.mp4'.format(os.path.join(path,file)))

os.system('mv *labeled.mp4 trim/')

os.chdir('/Users/claireeverett/Desktop/Process_input/videos/trim/')


for file in os.listdir():
        if file.endswith(".mp4"):
            print(file.split('.mp4')[0] + file.split('.mp4')[1])
            os.rename(file,file.split('_labeled.mp4')[0] + file.split('_labeled.mp4')[1] + '.mp4' )