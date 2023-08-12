# python libraries
import tarfile, os, warnings
from urllib.request import urlretrieve

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# local libraries
from .dataset_base import EnJaDataset


class JESC(EnJaDataset):
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
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{JESC.OUT_NAME}"
        )
        if not force_override and os.path.exists(output_path):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=JESC.OUT_NAME
                )
            )
            return
        JESC._download_raw()
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
        # create csv file
        with open(output_path, "wb+") as csv_file:
            header_str = EnJaDataset.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            with tarfile.open(
                f"{EnJaDataset.DATASET_RAW_DIR}/JESC/raw.tar.gz", mode="r"
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
        print(JESC.INFO)
        return
    
    @staticmethod
    def load():
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{JESC.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=JESC.OUT_NAME))
            JESC.create_csv()
            
        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data


    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{EnJaDataset.DATASET_RAW_DIR}/JESC"
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
                url=JESC.DOWNLOAD_URL,
                filename=output_path,
                reporthook=log_progress,
            )
        return
