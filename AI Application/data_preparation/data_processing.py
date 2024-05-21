from .data_generation import Data_Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import future  # for handling non-numeric frame rate
import time
import sys
from ffmpeg import input, output


data_generation = Data_Generation()

class Data_Processing:
    def __init__(self):
        pass

    def interview_bot_splitter(self, query):
        result = data_generation.generate_result(query)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_text(result)

        return doc_splits

    def store_docs_in_db(self, query):
        doc_splits = self.interview_bot_splitter(query)
        ollama_emb = OllamaEmbeddings(
                model="nomic-embed-text"
            )
        db = Chroma.from_texts(
            doc_splits,
            collection_name="rag-chroma",
            embedding=ollama_emb,
        )
        return db

    # Function for processing each frame and saving to video crewai
    def process_frames(input_directory = "data/crewai_input", fps = 25.0, output_directory = "data/crewai_output"):
        images = [f[0] for f in os.scandir(input_directory) if not f.is_dir()]
        total = len(images)

        # Determining frame rate from argument
        if isinstance(fps, int):
            fps = fps
        else:

            @future.wrap_future
            def delay(
                sec,
            ):  # Function for handling non-numeric frame rate using future library
                return sec

            delay_time = delay(1 / fps)

        out = output(
            output_directory + "/output.mp4", codec="libx264"
        )  # Opening output video file for writing with FFmpeg

        for i, image in enumerate(images):
            start_time = time.time()  # Recording processing time per frame
            frame = data_generation.read_image_and_convert_to_frame(os.path.join(input_directory, image))
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
            if isinstance(fps, int):
                time.sleep(1 / fps)  # Simple sleep function for integer frame rate
            else:
                time.sleep(
                    delay_time.seconds
                )  # Delayed sleep function for non-numeric frame rate

            print("\r" + "=" * 30)  # Clearing the console line after frame processing

        out.run()  # Write the final video output file
        print("\nVideo processing completed!")