# python libraries
import os

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar, concatenate_datasets

# local libraries
from .dataset_base import EnJaDataset


class OPUS100(EnJaDataset):
    OUT_NAME = r"opus100.csv"
    INFO = (
        "Webpage    : https://github.com/EdinburghNLP/opus-100-corpus\n"
        "Webpage(HF): https://huggingface.co/datasets/opus100\n"
        "Summary    : a multilingual corpus with 1M [en-ja] sentences,\n"
        "             of various origins."
    )

    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{EnJaDataset.DATASET_PROCESSED_DIR}/{OPUS100.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=OPUS100.OUT_NAME
                )
            )
            return
        dataset = load_dataset("opus100", "en-ja", cache_dir=EnJaDataset.DATASET_RAW_DIR)
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
        
        dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        
        disable_progress_bar()
        def rearrange(batch):
            batch["en_sentence"] = batch["translation"]["en"]
            batch["ja_sentence"] = batch["translation"]["ja"]
            return batch
        dataset = dataset.map(rearrange, remove_columns=["translation"], num_proc=EnJaDataset.NUM_PROC)
        def remove_jank(batch):
            # 1 -> remove jank, 2 -> remove more jank, 3 -> ???
            # japanese has None or empty string...
            # jank keeps increasing "n/a" in Japanese ("n/ a" in en) gets loaded as None
            return len(batch["en_sentence"]) > 2 and len(batch["ja_sentence"]) > 0 \
                and batch["ja_sentence"] not in ["n/a", "N/A", "なし"]
        
        dataset = dataset.filter(remove_jank)
        
        dataset.to_csv(output_path)
        enable_progress_bar()
        return
        

    @staticmethod
    def info():
        print(OPUS100.INFO)
        return
    
    @staticmethod
    def load():
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{OPUS100.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=OPUS100.OUT_NAME))
            OPUS100.create_csv()

        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data