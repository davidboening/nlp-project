# python libraries
import os

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar, concatenate_datasets

# local libraries
from .dataset_base import EnJaDataset


class IWSLT2017(EnJaDataset):
    OUT_NAME = r"iwslt2017.csv"
    INFO = (
        "Webpage    : https://sites.google.com/site/iwsltevaluation2017/TED-tasks\n"
        "Webpage(HF): https://huggingface.co/datasets/iwslt2017\n"
        "Summary    : a collection of multilingual tasks, one of which is a bilingual\n"
        "             corpus of 230k [en-ja] sentences."
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{EnJaDataset.DATASET_PROCESSED_DIR}/{IWSLT2017.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=IWSLT2017.OUT_NAME
                )
            )
            return
        dataset = load_dataset("iwslt2017", 'iwslt2017-en-ja', cache_dir=EnJaDataset.DATASET_RAW_DIR)
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
        
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        
        disable_progress_bar()
        def rearrange(batch):
            batch["en_sentence"] = batch["translation"]["en"]
            batch["ja_sentence"] = batch["translation"]["ja"]
            return batch
        dataset = dataset.map(rearrange, remove_columns=["translation"], num_proc=EnJaDataset.NUM_PROC)
        
        dataset.to_csv(output_path)
        enable_progress_bar()
        return
        

    @staticmethod
    def info():
        print(IWSLT2017.INFO)
        return
    
    @staticmethod
    def load():
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{IWSLT2017.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=IWSLT2017.OUT_NAME))
            IWSLT2017.create_csv()

        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data