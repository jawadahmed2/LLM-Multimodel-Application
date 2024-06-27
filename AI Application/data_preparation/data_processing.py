from .data_generation import Data_Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
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
import json
from loguru import logger


data_generation = Data_Generation()

class Data_Processing:
    def __init__(self):
        pass

    def interview_bot_splitter(self, query):
        """
        Split text obtained from an interview bot.
        """
        result = data_generation.generate_result(query)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_text(result)

        return doc_splits

    def image_processing(self, inputs: dict) -> dict:
        """
        Load image from file and encode it as base64.
        """
        image_path = inputs["image_path"]
        pil_image = Image.open(image_path)
        image_base64 = data_generation.generate_base64_image(pil_image)
        return {"image": image_base64}


    def process_instructions(self, db, chain, query) -> None:
        """
        Process instructions by invoking a chain for each document in the database.
        """
        vs = db.__dict__.get("docstore")
        docstore_id_list = list(db.__dict__.get("index_to_docstore_id").values())
        rand_doc_id_list = random.choices(docstore_id_list, k=100)

        qfile = open("data_preparation/data/llm_tuning/instructions.txt", "w")
        start_gen = timeit.default_timer()
        for i, doc_id in enumerate(rand_doc_id_list):
            start = timeit.default_timer()
            a_doc = vs.search(doc_id)
            result = chain.invoke({"question": query, "context": a_doc.page_content})
            resp_time = timeit.default_timer() - start
            logger.info(f'{"-"*50}\nQ #{i}: {result}\nTime: {resp_time}\n{"-"*50}\n')
            qfile.write(result[3:])
        qfile.close()
        gen_time = timeit.default_timer() - start_gen
        logger.info(f"Total generation time => {timedelta(seconds=gen_time)}")

    def process_training(self, db, bm25_r, chain,) -> None:
        """
        Process training data by invoking a chain for each instruction.
        """
        with open("data_preparation/data/llm_tuning/instructions.txt") as tfile:
            instructions = tfile.readlines()
        start_t_gen = timeit.default_timer()
        train_lines = list()
        for i, instruction in enumerate(instructions, start=1):
            logger.info(f"Handling ({i}/{len(instructions)}):")
            start = timeit.default_timer()
            try:
                answer = chain.invoke(instruction)
            except Exception as e:
                logger.debug(f"FAILED for => {e}")
                continue
            resp_time = timeit.default_timer() - start
            logger.info(f'{"-"*50}\nQ #{i}: {instruction}\nA:{answer}\nTime: {resp_time}\n{"-"*50}\n')
            result = json.dumps({"text": f"<s>[INST] {instruction}[/INST] {answer}</s>"}) + "\n"
            with open("data_preparation/data/llm_tuning/train_valid.jsonl", "a") as file:
                file.write(result)
            train_lines.append(result)
        gen_time = timeit.default_timer() - start_t_gen
        with open("data_preparation/data/llm_tuning/valid.jsonl", "w") as file:
            file.writelines(train_lines[: int(len(train_lines) * 0.2)])
        with open("data_preparation/data/llm_tuning/train.jsonl", "w") as file:
            file.writelines(train_lines[int(len(train_lines) * 0.2) :])
        logger.info(f"Total training generation time => {timedelta(seconds=gen_time)}")


    def process_df2Graph(self, dataframe: pd.DataFrame, graphPrompt_result) -> list:
        """
        Process DataFrame to extract a list of entities.
        """
        results = graphPrompt_result.dropna().reset_index(drop=True)
        concept_list = np.concatenate(results).ravel().tolist()
        return concept_list

    def process_graph2Df(self, nodes_list) -> pd.DataFrame:
        """
        Process a list of nodes into a DataFrame.
        """
        graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
        graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
        graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
        graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
        return graph_dataframe

    def proces_contextual_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate contextual proximity within a DataFrame.
        """
        dfg_long = pd.melt(
            df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
        )
        dfg_long.drop(columns=["variable"], inplace=True)
        dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
        self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
        dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
        dfg2 = (
            dfg2.groupby(["node_1", "node_2"])
            .agg({"chunk_id": [",".join, "count"]})
            .reset_index()
        )
        dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
        dfg2.replace("", np.nan, inplace=True)
        dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
        dfg2 = dfg2[dfg2["count"] != 1]
        dfg2["edge"] = "contextual proximity"
        return dfg2
