# external libraries
from datasets import load_dataset

# python libraries
import os
import itertools

# local libraries
from .settings import DatasetConfig


class MassiveTranslationDataset:
    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{DatasetConfig.DATASET_PROCESSED_DIR}/{DatasetConfig.MASSIVE_TRANSLATION_OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                DatasetConfig.SKIPPED_MSG_FORMAT.format(
                    file=DatasetConfig.MASSIVE_TRANSLATION_OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "Amani27/massive_translation_dataset",
            cache_dir=DatasetConfig.DATASET_RAW_DIR,
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
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for en_s, ja_s in zip(en_text, ja_text):
                out_line = f'"{en_s}","{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(DatasetConfig.MASSIVE_TRANSLATION_INFO)
