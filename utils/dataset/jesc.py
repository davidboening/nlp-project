# python libraries
import tarfile, os, warnings
from urllib.request import urlretrieve

# external libraries
from datasets import load_dataset
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# local libraries
from .dataset_loader import DatasetLoader


class JESCDataset(DatasetLoader):
    DOWNLOAD_URL = r"https://nlp.stanford.edu/projects/jesc/data/raw.tar.gz"
    OUT_NAME = r"jesc.csv"
    INFO = (
        "Webpage: https://nlp.stanford.edu/projects/jesc/\n"
        "Paper  : https://arxiv.org/abs/1710.10639\n"
        "Summary: Japanese-English Subtitle Corpus (2.8M sentences)"
    )

    @staticmethod
    def create_csv(force_override=False):
        # check processed file presence
        output_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{JESCDataset.OUT_NAME}"
        )
        if not force_override and os.path.exists(output_path):
            print(
                DatasetLoader.SKIPPED_MSG_FORMAT.format(
                    file=JESCDataset.OUT_NAME
                )
            )
            return
        JESCDataset._download_raw()
        if not os.path.exists(DatasetLoader.DATASET_PROCESSED_DIR):
            os.makedirs(DatasetLoader.DATASET_PROCESSED_DIR)
        # create csv file
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetLoader.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            with tarfile.open(
                f"{DatasetLoader.DATASET_RAW_DIR}/JESC/raw.tar.gz", mode="r"
            ) as tfh:
                with tfh.extractfile("raw/raw") as fh:
                    while line := fh.readline():
                        line = line.decode().replace('"', '""')
                        sep = line.find("\t")
                        en_s, jp_s = line[:sep], line[sep + 1 : -1]
                        out_line = f'"{en_s}","{jp_s}"\n'
                        csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(JESCDataset.INFO)
        return

    @staticmethod
    def stats(en_tokenizer, ja_tokenizer, num_proc=4):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{JESCDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=JESCDataset.OUT_NAME))
            return
        DatasetLoader.stats(
            csv_path, 
            en_tokenizer=en_tokenizer, 
            ja_tokenizer=ja_tokenizer, 
            num_proc=num_proc
        )
        return
    
    @staticmethod
    def load(**kwargs):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{JESCDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=JESCDataset.OUT_NAME))
            return
        return load_dataset("csv", data_files=csv_path, **kwargs)


    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{DatasetLoader.DATASET_RAW_DIR}/JESC"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = f"{output_dir}/raw.tar.gz"
        if force_download or not os.path.exists(output_path):
            progress_bar = None

            def log_progress(c, s, t):
                nonlocal progress_bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        desc="Downloading Dataset",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1000,
                    )
                progress_bar.update(s)

            urlretrieve(
                url=JESCDataset.DOWNLOAD_URL,
                filename=output_path,
                reporthook=log_progress,
            )
        return
