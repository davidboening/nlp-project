# external libraries
from datasets import load_dataset
# python libraries
import os

class SnowSimplifiedDataset:
    def __init__(self,
        cache_dir=r"./data", 
        output_path=r"./data-post/snow_simplified.csv"
    ):
        self.dataset_path = cache_dir
        self.output_path = output_path

    def create_csv(self):
        if os.path.exists(self.output_path):
            print(f"skipped: snow_simplified file already exists!")
            return
        dataset = load_dataset("snow_simplified_japanese_corpus", cache_dir=self.dataset_path)
        with open(self.output_path, "wb+") as csv_file:
            header_str = f'en_sentence, ja_sentence\n'
            csv_file.write(header_str.encode("utf-8"))
            for rec in dataset["train"]:
                en_s, ja1_s, ja2_s = rec["original_en"], rec["original_ja"], rec["simplified_ja"]
                out_line = f'"{en_s}", "{ja1_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
                out_line = f'"{en_s}", "{ja2_s}"\n'
                csv_file.write(out_line.encode("utf-8"))
        return