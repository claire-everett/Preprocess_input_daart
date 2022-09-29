#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:58:47 2022

@author: claireeverett
"""

#pip install --upgrade tables

import pickle
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from glob import glob
from scipy import signal

os.chdir('/Users/claireeverett/Desktop/Process_input/scripts/')


from helper_functions_VA import *
#%%

# load the trials information

meta_dir = '/Users/claireeverett/Desktop/Process_input/contour_info/'

os.chdir(meta_dir)

base_features = []
contours_paths = []
tail_indexs = []
tail_angle_paths = []
tail_dev_paths = []


for dir in next(os.walk('.'))[1]:
    base_features.append(meta_dir + dir + '/base_features.csv')
    contours_paths.append(meta_dir + dir + '/curve_scores')
    tail_indexs.append(meta_dir + dir + '/tail_index')
    tail_angle_paths.append(meta_dir + dir + '/tailAngle')
    tail_dev_paths.append(meta_dir + dir + '/tail_dev')
    

DLC_dir = '/Users/claireeverett/Desktop/Process_input/h5/'
DLC_paths = sorted(glob(os.path.join(DLC_dir ,'*.h5')))
#%%
def find_centroid(contour):
    moments=cv2.moments(contour)
    m01=moments.get("m01")
    m10=moments.get("m10")
    m00=moments.get("m00")
    xbar=m10/m00
    ybar=m01/m00
    return xbar,ybar

def concat_features(base_features, DLC_path,contours_path,tail_index_path,tail_angle_path,tail_dev_path):
  fish_name = os.path.basename(os.path.dirname(base_features))
  print(fish_name)
  base_df = pd.read_csv(base_features)
  path = DLC_path
  f = pd.HDFStore(path,'r')
  df = f.get('df_with_missing')
  df.columns = df.columns.droplevel()
  head_x_pos,head_y_pos = df['A_head']['x'],df['A_head']['y']
  head_x_pos = np.array(head_x_pos.iloc[START_FRAME:END_FRAME])
  head_y_pos = np.array(head_y_pos.iloc[START_FRAME:END_FRAME])
  base_df['head_x'] = head_x_pos
  base_df['head_y'] = head_y_pos
  with open(contours_path,"rb") as fp:
    contours = pickle.load(fp)
  with open(tail_index_path,"rb") as fp:
    tail_indexs = pickle.load(fp)
  centroid_x = []
  centroid_y = []
  tail_x = []
  tail_y = []
  for tail_index,contour in zip(tail_indexs,contours):
    t_x,t_y = contour[tail_index%len(contour),:2]
    contour = np.array(contour[:,:2][:,None,:],dtype = np.int)
    xbar,ybar = find_centroid(contour)
    centroid_x.append(xbar)
    centroid_y.append(ybar)
    tail_x.append(t_x)
    tail_y.append(t_y)
  base_df['centroid_x'] = centroid_x
  base_df['centroid_y'] = centroid_y
  base_df['tail_x'] = tail_x
  base_df['tail_y'] = tail_y
  with open(tail_angle_path,"rb") as fp:
    tail_angle = pickle.load(fp)
  base_df['tail_angle'] = tail_angle
  with open(tail_dev_path,"rb") as fp:
    tail_dev = pickle.load(fp)
  base_df['tail_dev'] = tail_dev

  if 'L' in fish_name:
      print("flipping coordinate for {}".format(fish_name))
      base_df['orientation'] = 180-base_df['orientation']
      base_df['head_x'] = 500-base_df['head_x']
      base_df['centroid_x'] = 500-base_df['centroid_x']
      base_df['tail_x'] = 500-base_df['tail_x']
  return base_df

START_FRAME = 72000
END_FRAME = 144000

for i in tqdm(range(len(base_features))):
  new_df = concat_features(base_features[i],DLC_paths[i],contours_paths[i],tail_indexs[i],tail_angle_paths[i],tail_dev_paths[i])
  new_df.to_csv(os.path.join(os.path.dirname(base_features[i]), 'data_auto_scored.csv'))
  
#%%
# reload the temporary basefeatures to add more and to make filter
new_base_features = []

for dir in next(os.walk('.'))[1]:
    new_base_features.append(meta_dir + dir + '/data_auto_scored.csv')

export_dir = '/Users/claireeverett/Desktop/Process_input/basefeatures/'

#%%
for i in np.arange(len(DLC_paths)):
    # if os.path.basename(base_features[i]).split('_')[0] == os.path.basename(DLC_paths[i]).split('_')[0]:
    #     if os.path.basename(bf_files[i]).split('_')[1][5] == os.path.basename(h5_files[i]).split('_')[1][5]:
            
    # load the .h5 files 
    file_handle_auto = DLC_paths[i]
    
    with pd.HDFStore(file_handle_auto,'r') as file:
        data_auto = file.get('df_with_missing')
        data_auto.columns= data_auto.columns.droplevel()
    
    # load the basefeatures
    base_new = pd.read_csv(new_base_features[i])
    
    new_features=features(starttime=0, duration=len(data_auto))
    filtered_df=new_features.filter_df(data_auto)
    new_features.fit(filtered_df,filter_feature=True,fill_na=True,estimate_na=False) 
    oper_angle = new_features.operculum

    # for each file in list, make a dataframe of different operculum features
    operangle_R = auto_scoring_get_angle(filtered_df, ['A_head', 'B_rightoperculum', 'F_spine1'])[72000:144000]
    operangle_L = auto_scoring_get_angle(filtered_df, ['A_head', 'E_leftoperculum', 'F_spine1'])[72000:144000]
    
    operdist_R = oper_diff(filtered_df, ['B_rightoperculum', 'F_spine1'])[72000:144000]
    operdist_L = oper_diff(filtered_df, ['E_leftoperculum', 'F_spine1'])[72000:144000]
    
    oper_avg = (operangle_R + operangle_L)/2
    
    dist_avg = (operdist_R + operdist_L)/2
    
    base_new['oper_angle_R'] = np.array(operangle_R )
    base_new['oper_angle_L'] = np.array(operangle_L)
    base_new['oper_dist_R'] = np.array(operdist_R)
    base_new['oper_dist_L'] = np.array(operdist_R)
    base_new['oper_angle_avg'] = np.array(oper_avg)
    base_new['oper_dist_avg'] = np.array(dist_avg)
    
    base_new = base_new.drop('Unnamed: 0', axis = 1)
    base_new = base_new.fillna(method = 'ffill')
    base_new = base_new.fillna(method = 'bfill')
    
    df_filt = pd.DataFrame(index = base_new.index, columns = base_new.columns)
    for p in base_new.columns:
        df_filt[p] = signal.medfilt(base_new[p], 11)
    
    df_filt.to_csv(os.path.join(export_dir,os.path.basename(DLC_paths[i]).split('DLC')[0] + '_labeled.csv' ))
    



new_files = sorted(glob(os.path.join(export_dir,'*.csv')))



export_dir = '/Users/claireeverett/Desktop/Process_input/for_daart/'
for i in new_files:
    df = pd.read_csv(i, index_col=[0]) 
    df_filt = pd.DataFrame(index = df.index, columns = df.columns)
    for p in df.columns:
        df_filt[p] = signal.medfilt(df[p], 11)
        
    df_filt.to_csv(os.path.join(export_dir, os.path.basename(i)))
        
        
        
        
        
        