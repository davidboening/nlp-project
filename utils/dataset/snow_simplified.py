# python libraries
import os

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar, concatenate_datasets

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
        
        disable_progress_bar()
        def ogify(batch):
            batch["en_sentence"] = batch["original_en"]
            batch["ja_sentence"] = batch["original_ja"]
            return batch

        def simpfy(batch):
            batch["en_sentence"] = batch["original_en"]
            batch["ja_sentence"] = batch["simplified_ja"]
            return batch

        dataset_og = dataset.map(
            ogify, batch_size=1, 
            remove_columns=["ID", "original_ja", "original_en", "simplified_ja"]
        )
        dataset_simp = dataset.map(
            simpfy, batch_size=1, 
            remove_columns=["ID", "original_ja", "original_en", "simplified_ja"]
        )
        dataset = concatenate_datasets([dataset_og["train"], dataset_simp["train"]])
        dataset.to_csv(output_path)
        enable_progress_bar()
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
