# python libraries
import os

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar

# local libraries
from .dataset_base import EnJaDataset


class SnowSimplified(EnJaDataset):
    OUT_NAME = r"snow_simplified.csv"
    INFO = (
        "Webpage: https://huggingface.co/datasets/snow_simplified_japanese_corpus\n"
        "Summary: Japanese-English sentence pairs, all Japanese sentences have\n"
        "         a simplified counterpart (85k(x2) sentences)"
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{EnJaDataset.DATASET_PROCESSED_DIR}/{SnowSimplified.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=SnowSimplified.OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "snow_simplified_japanese_corpus", cache_dir=EnJaDataset.DATASET_RAW_DIR
        )
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
        with open(output_path, "wb+") as csv_file:
            header_str = EnJaDataset.CSV_HEADER_STR
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
        print(SnowSimplified.INFO)
        return
    
    @staticmethod
    def load():
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{SnowSimplified.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=SnowSimplified.OUT_NAME))
            SnowSimplified.create_csv()

        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data
