Video Segmentation
This project provides tools for segmenting  videos using the Segment Anything Model (SAM) and a Tkinter-based GUI. It also includes scripts for cutting videos into segments.

Requirements
Install the required Python packages using pip:

```
pip install -r requirements.txt
```
And then download the model from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints


Usage
Cutting Videos into Segments
To cut a video into 2-minute segments, use the following ffmpeg command:
```
ffmpeg -i <your_video>.mp4 -c copy -map 0 -segment_time 120 -f segment -reset_timestamps 1 <your_video>_%03d.mp4
```

This command will split <your_video>.mp4 into multiple segments, each 2 minutes long, and save them as <your_video>_000.mp4, <your_video>_001.mp4, etc.



Segmenting Videos with SAM
To run the Tkinter-based GUI for segmenting videos, run:
python video.py
