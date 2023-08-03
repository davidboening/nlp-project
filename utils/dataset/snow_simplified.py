# external libraries
from datasets import load_dataset
# python libraries
import os
# local libraries
from .settings import DatasetConfig

class SnowSimplifiedDataset:
    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{DatasetConfig.PROCESSED_DATA_DIR}/{DatasetConfig.SNOW_SIMPLIFIED_OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(DatasetConfig.SKIPPED_MSG_FORMAT.format(file=DatasetConfig.SNOW_SIMPLIFIED_OUT_NAME))
            return
        dataset = load_dataset("snow_simplified_japanese_corpus", cache_dir=DatasetConfig.HF_DATASET_RAW_DIR)
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for rec in dataset["train"]:
                en_s, ja1_s, ja2_s = rec["original_en"], rec["original_ja"], rec["simplified_ja"]
                out_line = f'"{en_s}", "{ja1_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
                out_line = f'"{en_s}", "{ja2_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return
    @staticmethod
    def info():
        print(DatasetConfig.SNOW_SIMPLIFIED_INFO)