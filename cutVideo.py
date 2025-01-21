ffmpeg -i basketball.mp4 -c copy -map 0 -segment_time 120 -f segment -reset_timestamps 1 basketball_%03d.mp4
