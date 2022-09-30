#Preprocess_input_daart
### activate daart environment
```
conda activate daart
```
### navigate to the Process_input folder in terminal
```
cd ~/Desktop/GitHub/Preprocess_input_daart/
```
## Convert .h264 files to .mp4, crop videos, and concat together

### Step 1: Add all videos to Process_input>videos
### Step 2: Add cropping coordinates for each file in test_coordinates.csv
### Step 3: Run cropping script
```
python scripts/crop_videos.py "lesion3"
```
### Step 4: Remove all folders containing .h264 (to save space and keep things organized)

```
cd videos/
rm -r -- ./*/
cd ..
```
## Track bodyparts in DeepLabCut

### Step 1: Sign into axon and transfer files to correct folder in axon
```
ssh cpe2108@axon.rc.zi.columbia.edu
scp videos/*.mp4 cpe2108@axon.rc.zi.columbia.edu:~/top_Feb27_update/top_videos/
```
### Step 2: Run top track and label
```
sbatch ~/python_scripts/dlc_analyze_top_Feb27update.sh
```
### Step 3: Make labelled videos once the tracking is done and before you transfer files out of the top_videos folder
```
sbatch ~/python_scripts/dlc_label_Feb27.sh
```
### Step 4: Export .h5 and labelled videos into designated folders
```
scp cpe2108@axon.rc.zi.columbia.edu:~/top_Feb27_update/top_videos/*.h5 ./h5/
scp cpe2108@axon.rc.zi.columbia.edu:~/top_Feb27_update/top_videos/*labeled.mp4 ./labeled_videos/
```
### Step 5: watch labelled video and run .h5 through quality control script

## Trim video
### Trimming the video to only the test period allows for a faster contour step. You can do this while you wait for DLC. 

### Step 1: Run trimming script
```
python scripts/trim_videos.py
```
## Create Contour

### Step 1: Take the contour of fish, adjust kernel size and masking as needed
```
python scripts/contour_script_CE.py 
```
### Step 2: Check the efficacy of the contour: Watch sample output of the contour to check for skips/inaccuracies

### Step 3: Use the markers and contours to create basefeatures.csv (daart input)
```
python scripts/basefeatures_input_CE.py
```
***Make sure the left fish has ‘L’ in its name, otherwise the script won’t know to flip it at the end***

## Supervised Scoring of flaring using Daart


### Step 1: Input Basefeatures into daart
```
conda activate daart

cd scripts/

jupyter notebook betta_report.ipynb
```
### Step 2: click through the notebook to generate files

### Step 3: Find .npy of binary time series in Process_input>version_0 

### Step 4: Find daart validation metrices in Process_input>daart_output




