import tarfile
import os

class JESTDataset:
    def __init__(self,
        dataset_path=r"./data/raw.tar.gz", 
        output_path=r"./data-post/jest.csv"
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path

    def create_csv(self):
        if os.path.exists(self.output_path):
            print(f"skipped: JEST file already exists!")
            return
        with open(self.output_path, "wb+") as csv_file:
            header_str = f'en_sentence, ja_sentence\n'
            csv_file.write(header_str.encode("utf-8"))
            with tarfile.open(self.dataset_path, mode="r") as tfh:
                with tfh.extractfile("raw/raw") as fh:
                    while line := fh.readline():
                        line = line.decode()
                        sep = line.find("\t")
                        en_s, jp_s = line[:sep], line[sep+1:-1]
                        out_line = f'"{en_s}", "{jp_s}"\n'
                        csv_file.write(out_line.encode("utf-8"))
        return