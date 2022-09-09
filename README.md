# Preprocess_input_daart

### navigate to the Process_input folder in terminal

cd ~/Desktop/GitHub/Process_input/

## Convert .h264 files to .mp4, crop videos, and concat together

### Step 1: Add all videos to Process_input>videos
### Step 2: Add cropping coordinates for each file in test_coordinates.csv
### Step 3: Run cropping script

python /scripts/crop_videos.py "lesion3"

### Step 4: Remove all folders containing .h264 (to save space and keep things organized)

rm -r -- ./*/

## Track bodyparts in DeepLabCut

### sign into axon and transfer files to correct folder in axon

ssh cpe2108@axon.rc.zi.columbia.edu
scp /videos/*.mp4 cpe2108@axon.rc.zi.columbia.edu:~/top_FEb27_update/top_videos/

### Run top track and label

sbatch ~/python_scripts/dlc_analyze_top_Feb27update.sh

### make labelled videos once the tracking is done and before you transfer files out of the top_videos folder

sbatch ~/python_scripts/dlc_label_top.sh

### export .h5 and labelled videos into designated folders

scp cpe2108@axon.rc.zi.columbia.edu:~/top_FEb27_update/top_videos/*.h5 ./h5/
scp cpe2108@axon.rc.zi.columbia.edu:~/top_FEb27_update/top_videos/*labeled.mp4 ./labeled_videos/

### watch labelled video and run .h5 through quality control script


## Creating Contour

