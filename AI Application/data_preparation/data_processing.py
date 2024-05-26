from .data_generation import Data_Generation
from infographics.generate_report import Generate_Report
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import future  # for handling non-numeric frame rate
import time
import sys
from ffmpeg import input, output
from PIL import Image
import random
from datetime import timedelta
import timeit


data_generation = Data_Generation()
generate_report = Generate_Report()

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

    def image_processing(self, inputs: dict) -> dict:
        "Load image from file and encode it as base64."
        image_path = inputs["image_path"]
        pil_image = Image.open(image_path)
        image_base64 = data_generation.generate_base64_image(pil_image)
        generate_report.plt_img_base64(image_base64)
        return {"image": image_base64}



    def process_instructions(self,db, chain, query) -> None:
        # access vector for k-doc chunks
        vs = db.__dict__.get("docstore")
        docstore_id_list = list(db.__dict__.get("index_to_docstore_id").values())
        rand_doc_id_list = random.choices(docstore_id_list, k=100)

        qfile = open("data_preparation/data/llm_tuning/instructions.txt", "w")
        start_gen = timeit.default_timer()
        for i, doc_id in enumerate(rand_doc_id_list):
            start = timeit.default_timer()
            a_doc = vs.search(doc_id)
            # print(f'CHOSEN DOC => {a_doc.page_content}\n_________________\n')
            result = chain.invoke({"question": query, "context": a_doc.page_content})
            resp_time = timeit.default_timer() - start  # seconds
            print(f'{"-"*50}\nQ #{i}: {result}\nTime: {resp_time}\n{"-"*50}\n')
            qfile.write(result[3:])
        qfile.close()
        gen_time = timeit.default_timer() - start_gen  # seconds
        print(f"Total generation time => {timedelta(seconds=gen_time)}")

    def process_training(self, db, bm25_r, chain,) -> None:

        with open("data_preparation/data/llm_tuning/instructions.txt") as tfile:
            instructions = tfile.readlines()
        start_t_gen = timeit.default_timer()
        train_lines = list()
        for i, instruction in enumerate(instructions, start=1):
            print(f"Handling ({i}/{len(instructions)}):")
            start = timeit.default_timer()
            try:
                answer = chain.invoke(instruction)
            except Exception as e:
                print(f"FAILED for => {e}")
                continue
            resp_time = timeit.default_timer() - start  # seconds
            print(
                f'{"-"*50}\nQ #{i}: {instruction}\nA:{answer}\nTime: {resp_time}\n{"-"*50}\n'
            )
            result = (
                json.dumps({"text": f"<s>[INST] {instruction}[/INST] {answer}</s>"}) + "\n"
            )
            with open("data_preparation/data/llm_tuning/train_valid.jsonl", "a") as file:
                file.write(result)
            train_lines.append(result)
        gen_time = timeit.default_timer() - start_t_gen  # seconds
        with open("data_preparation/data/llm_tuning/valid.jsonl", "w") as file:
            file.writelines(train_lines[: int(len(train_lines) * 0.2)])
        with open("data_preparation/data/llm_tuning/train.jsonl", "w") as file:
            file.writelines(train_lines[int(len(train_lines) * 0.2) :])
        print(f"Total training generation time => {timedelta(seconds=gen_time)}")


    def process_df2Graph(self, dataframe: pd.DataFrame, graphPrompt_result) -> list:
        # invalid json results in NaN
        results = graphPrompt_result.dropna()
        results = results.reset_index(drop=True)

        ## Flatten the list of lists to one single list of entities.
        concept_list = np.concatenate(results).ravel().tolist()
        return concept_list

    def process_graph2Df(self, nodes_list) -> pd.DataFrame:
        ## Remove all NaN entities
        graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
        graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
        graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
        graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
        return graph_dataframe

    def proces_contextual_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        ## Melt the dataframe into a list of nodes
        dfg_long = pd.melt(
            df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
        )
        dfg_long.drop(columns=["variable"], inplace=True)
        # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
        dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
        # drop self loops
        self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
        dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
        ## Group and count edges.
        dfg2 = (
            dfg2.groupby(["node_1", "node_2"])
            .agg({"chunk_id": [",".join, "count"]})
            .reset_index()
        )
        dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
        dfg2.replace("", np.nan, inplace=True)
        dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
        # Drop edges with 1 count
        dfg2 = dfg2[dfg2["count"] != 1]
        dfg2["edge"] = "contextual proximity"
        return dfg2
