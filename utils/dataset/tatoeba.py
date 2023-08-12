# python libraries
import os

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar

# local libraries
from .dataset_base import EnJaDataset


class Tatoeba(EnJaDataset):
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
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{Tatoeba.OUT_NAME}"
        )
        if not force_override and os.path.exists(output_path):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=Tatoeba.OUT_NAME
                )
            )
            return
        dataset = load_dataset(
            "tatoeba", lang1="en", lang2="ja", cache_dir=EnJaDataset.DATASET_RAW_DIR
        )
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)

        with open(output_path, "wb+") as csv_file:
            header_str = EnJaDataset.CSV_HEADER_STR
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
        print(Tatoeba.INFO)
        return
    
    @staticmethod
    def load():
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{Tatoeba.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=Tatoeba.OUT_NAME))
            Tatoeba.create_csv()

        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data
