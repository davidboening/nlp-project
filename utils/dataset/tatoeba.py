# external libraries
from datasets import load_dataset
# python libraries
import os
# local libraries
from .settings import DatasetConfig

class TatoebaDataset:
    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{DatasetConfig.DATASET_PROCESSED_DIR}/{DatasetConfig.TATOEBA_OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(DatasetConfig.SKIPPED_MSG_FORMAT.format(file=DatasetConfig.TATOEBA_OUT_NAME))
            return
        dataset = load_dataset("tatoeba", lang1="en", lang2="ja", cache_dir=DatasetConfig.DATASET_RAW_DIR)
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for rec in dataset["train"]["translation"]:
                en_s, ja_s = rec["en"], rec["ja"]
                out_line = f'"{en_s}","{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return
    @staticmethod
    def info():
        print(DatasetConfig.TATOEBA_INFO)