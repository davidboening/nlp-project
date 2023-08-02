# external libraries
from datasets import load_dataset
# python libraries
import os
import itertools

class MassiveTranslationDataset:
    def __init__(self,
        cache_dir=r"./data", 
        output_path=r"./data-post/massive_translation.csv"
    ):
        self.dataset_path = cache_dir
        self.output_path = output_path

    def create_csv(self):
        if os.path.exists(self.output_path):
            print(f"skipped: massive_translation file already exists!")
            return
        dataset = load_dataset("Amani27/massive_translation_dataset", cache_dir=self.dataset_path)
        en_text = itertools.chain(dataset["train"]["en_US"], dataset["validation"]["en_US"], dataset["test"]["en_US"])
        ja_text = itertools.chain(dataset["train"]["ja_JP"], dataset["validation"]["ja_JP"], dataset["test"]["ja_JP"])
        with open(self.output_path, "wb+") as csv_file:
            header_str = f'en_sentence, ja_sentence\n'
            csv_file.write(header_str.encode("utf-8"))
            for (en_s, ja_s) in zip(en_text, ja_text):
                out_line = f'"{en_s}", "{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return