# output.py
# Importing necessary libraries
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import future  # for handling non-numeric frame rate
import time
import sys
import argparse
from ffmpeg import input, output

# Set up command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "input_directory", type=str, help="Input directory containing image files"
)
parser.add_argument(
    "output_directory", type=str, help="Output directory for resulting video file"
)
parser.add_argument(
    "--fps", default=25.0, type=float, help="Frame Rate per second (default: 25)"
)
args = parser.parse_args()

# Checking input and output directories existence
if not os.path.exists(args.input_directory):
    print("Input directory does not exist!")
    sys.exit()

if not os.path.exists(args.output_directory):
    os.makedirs(args.output_directory)


# Function for reading image file from input directory and converting to frame
def read_image_and_convert_to_frame(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Open image using OpenCV
    return np.asarray(image)  # Convert the OpenCV matrix to NumPy array for processing


# Function for processing each frame and saving to video
def process_frames(input_directory):
    images = [f[0] for f in os.scandir(input_directory) if not f.is_dir()]
    total = len(images)

    # Determining frame rate from argument
    if isinstance(args.fps, int):
        fps = args.fps
    else:

        @future.wrap_future
        def delay(
            sec,
        ):  # Function for handling non-numeric frame rate using future library
            return sec

        delay_time = delay(1 / args.fps)

    out = output(
        args.output_directory + "/output.mp4", codec="libx264"
    )  # Opening output video file for writing with FFmpeg

    for i, image in enumerate(images):
        start_time = time.time()  # Recording processing time per frame
        frame = read_image_and_convert_to_frame(os.path.join(input_directory, image))
        out.video.new_frame(frame)  # Write frame to output video file with FFmpeg

        if not (i % 50 == 0):  # Progress updates for every 50 frames using tqdm library
            continue

        percentage = round(
            (i / total * 100), 2
        )  # Calculating and printing frame processing percentage
        tqdm.write(
            f"\rProcessing frame {percentage}% ({i + 1}/{total})"
        )  # Writing frame processing percentage to console using progress bar from tqdm library

        # Pausing for determined frame rate before processing the next image
        if isinstance(args.fps, int):
            time.sleep(1 / args.fps)  # Simple sleep function for integer frame rate
        else:
            time.sleep(
                delay_time.seconds
            )  # Delayed sleep function for non-numeric frame rate

        print("\r" + "=" * 30)  # Clearing the console line after frame processing

    out.run()  # Write the final video output file
    print("\nVideo processing completed!")


# Calling process_frames function for image-to-video conversion
process_frames(args.input_directory)
