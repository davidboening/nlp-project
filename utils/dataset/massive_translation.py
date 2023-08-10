# external libraries
from datasets import load_dataset

# python libraries
import os
import itertools

# local libraries
from .dataset_loader import DatasetLoader


class MassiveTranslationDataset(DatasetLoader):
    OUT_NAME = r"massive_translation.csv"
    INFO = (
        "Webpage: https://huggingface.co/datasets/Amani27/massive_translation_dataset\n"
        "Summary: dataset derived from AmazonScience/MASSIVE for translation\n"
        "         (16k sentences in 10 languages)"
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{DatasetLoader.DATASET_PROCESSED_DIR}/{MassiveTranslationDataset.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                DatasetLoader.SKIPPED_MSG_FORMAT.format(
                    file=MassiveTranslationDataset.OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "Amani27/massive_translation_dataset",
            cache_dir=DatasetLoader.DATASET_RAW_DIR,
        )
        en_text = itertools.chain(
            dataset["train"]["en_US"],
            dataset["validation"]["en_US"],
            dataset["test"]["en_US"],
        )
        ja_text = itertools.chain(
            dataset["train"]["ja_JP"],
            dataset["validation"]["ja_JP"],
            dataset["test"]["ja_JP"],
        )
        if not os.path.exists(DatasetLoader.DATASET_PROCESSED_DIR):
            os.makedirs(DatasetLoader.DATASET_PROCESSED_DIR)
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetLoader.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for en_s, ja_s in zip(en_text, ja_text):
                out_line = f'"{en_s}","{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(MassiveTranslationDataset.INFO)
        return
    
    @staticmethod
    def stats(en_tokenizer, ja_tokenizer, num_proc=4):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{MassiveTranslationDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=MassiveTranslationDataset.OUT_NAME))
            return
        DatasetLoader.stats(
            csv_path, 
            en_tokenizer=en_tokenizer, 
            ja_tokenizer=ja_tokenizer, 
            num_proc=num_proc
        )
        return
