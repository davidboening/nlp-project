# python libraries
import os

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar, concatenate_datasets

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
        
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        
        disable_progress_bar()
        dataset = dataset.rename_columns({"en_US" : "en_sentence", "ja_JP" : "ja_sentence"})
        dataset.to_csv(output_path)
        enable_progress_bar()
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
