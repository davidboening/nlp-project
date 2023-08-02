# external libraries
from datasets import load_dataset
# python libraries
import os

class TatoebaDataset:
    def __init__(self,
        cache_dir=r"./data", 
        output_path=r"./data-post/tatoeba.csv"
    ):
        self.dataset_path = cache_dir
        self.output_path = output_path

    def create_csv(self):
        if os.path.exists(self.output_path):
            print(f"skipped: tatoeba file already exists!")
            return
        dataset = load_dataset("tatoeba", lang1="en", lang2="ja", cache_dir=self.dataset_path)
        with open(self.output_path, "wb+") as csv_file:
            header_str = f'en_sentence, ja_sentence\n'
            csv_file.write(header_str.encode("utf-8"))
            for rec in dataset["train"]["translation"]:
                en_s, ja_s = rec["en"], rec["ja"]
                out_line = f'"{en_s}", "{ja_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return