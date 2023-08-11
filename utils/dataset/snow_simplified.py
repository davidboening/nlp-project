# external libraries
from datasets import load_dataset

# python libraries
import os

# local libraries
from .dataset_loader import DatasetLoader


class SnowSimplifiedDataset:
    OUT_NAME = r"snow_simplified.csv"
    INFO = (
        "Webpage: https://huggingface.co/datasets/snow_simplified_japanese_corpus\n"
        "Summary: Japanese-English sentence pairs, all Japanese sentences have\n"
        "         a simplified counterpart (85k(x2) sentences)"
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{DatasetLoader.DATASET_PROCESSED_DIR}/{SnowSimplifiedDataset.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                DatasetLoader.SKIPPED_MSG_FORMAT.format(
                    file=SnowSimplifiedDataset.OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "snow_simplified_japanese_corpus", cache_dir=DatasetLoader.DATASET_RAW_DIR
        )
        if not os.path.exists(DatasetLoader.DATASET_PROCESSED_DIR):
            os.makedirs(DatasetLoader.DATASET_PROCESSED_DIR)
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetLoader.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for rec in dataset["train"]:
                en_s, ja1_s, ja2_s = (
                    rec["original_en"],
                    rec["original_ja"],
                    rec["simplified_ja"],
                )
                out_line = f'"{en_s}","{ja1_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
                out_line = f'"{en_s}","{ja2_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(SnowSimplifiedDataset.INFO)
        return
    
    @staticmethod
    def stats(en_tokenizer, ja_tokenizer, num_proc=4):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{SnowSimplifiedDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=SnowSimplifiedDataset.OUT_NAME))
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
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{SnowSimplifiedDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=SnowSimplifiedDataset.OUT_NAME))
            return
        return load_dataset("csv", data_files=csv_path, **kwargs)
