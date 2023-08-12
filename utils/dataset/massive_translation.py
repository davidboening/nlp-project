# python libraries
import os
import itertools

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar

# local libraries
from .dataset_base import EnJaDataset


class MassiveTranslation(EnJaDataset):
    OUT_NAME = r"massive_translation.csv"
    INFO = (
        "Webpage: https://huggingface.co/datasets/Amani27/massive_translation_dataset\n"
        "Summary: dataset derived from AmazonScience/MASSIVE for translation\n"
        "         (16k sentences in 10 languages)"
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{EnJaDataset.DATASET_PROCESSED_DIR}/{MassiveTranslation.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=MassiveTranslation.OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "Amani27/massive_translation_dataset",
            cache_dir=EnJaDataset.DATASET_RAW_DIR,
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
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
        with open(output_path, "wb+") as csv_file:
            header_str = EnJaDataset.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for en_s, ja_s in zip(en_text, ja_text):
                out_line = f'"{en_s}","{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(MassiveTranslation.INFO)
        return

    @staticmethod
    def load():
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{MassiveTranslation.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=MassiveTranslation.OUT_NAME))
            MassiveTranslation.create_csv()
            
        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data
