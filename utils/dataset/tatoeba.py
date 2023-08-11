# external libraries
from datasets import load_dataset

# python libraries
import os

# local libraries
from .dataset_loader import DatasetLoader


class TatoebaDataset:
    OUT_NAME = r"tatoeba.csv"
    INFO = (
        "Webpage    : https://opus.nlpl.eu/Tatoeba.php\nWebpage(HF):"
        " https://huggingface.co/datasets/tatoeba\nSummary    : a collection of"
        " sentences from https://tatoeba.org/en/, contains\n             over 400"
        " languages ([en-ja] 200k sentences)"
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{TatoebaDataset.OUT_NAME}"
        )
        if not force_override and os.path.exists(output_path):
            print(
                DatasetLoader.SKIPPED_MSG_FORMAT.format(
                    file=TatoebaDataset.OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "tatoeba", lang1="en", lang2="ja", cache_dir=DatasetLoader.DATASET_RAW_DIR
        )
        if not os.path.exists(DatasetLoader.DATASET_PROCESSED_DIR):
            os.makedirs(DatasetLoader.DATASET_PROCESSED_DIR)

        with open(output_path, "wb+") as csv_file:
            header_str = DatasetLoader.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for rec in dataset["train"]["translation"]:
                en_s, ja_s = rec["en"].replace('"', '""'), rec["ja"]
                if en_s == "" or ja_s == "":
                    continue
                out_line = f'"{en_s}","{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(TatoebaDataset.INFO)
        return
    
    @staticmethod
    def stats(en_tokenizer, ja_tokenizer, num_proc=4):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{TatoebaDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=TatoebaDataset.OUT_NAME))
            return
        DatasetLoader.stats(
            csv_path, 
            en_tokenizer=en_tokenizer, 
            ja_tokenizer=ja_tokenizer, 
            num_proc=num_proc
        )
        return
    
    @staticmethod
    def load(**kwargs):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{TatoebaDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=TatoebaDataset.OUT_NAME))
            return
        return load_dataset("csv", data_files=csv_path, **kwargs)
