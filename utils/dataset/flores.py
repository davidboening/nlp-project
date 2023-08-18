# python libraries
import tarfile, os, warnings
from urllib.request import urlretrieve

# external libraries
from datasets import load_dataset, enable_progress_bar, disable_progress_bar, Dataset
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# local libraries
from .dataset_base import EnJaDataset


class Flores(EnJaDataset):
    DOWNLOAD_URL = r"https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    OUT_NAMES = (r"flores.dev.csv", r"flores.devtest.csv")
    INFO = (
        "Webpage: https://github.com/facebookresearch/flores/tree/main/flores200\n"
        "Paper  : https://arxiv.org/abs/2207.04672\n"
        "Summary: Professional translation in over 200 languages, including\n"
        "         en-ja, for evaluation tasks."
    )

    @staticmethod
    def create_csv(force_override=False):
        # check processed file presence
        output_path1 = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{Flores.OUT_NAMES[0]}"
        )
        output_path2 = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{Flores.OUT_NAMES[1]}"
        )
        if not force_override and os.path.exists(output_path1) and os.path.exists(output_path2):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=Flores.OUT_NAMES
                )
            )
            return
        
        Flores._download_raw()
        
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
            
        # create csv file
        with tarfile.open(
        "./data-raw/flores/flores200_dataset.tar.gz", mode="r"
        ) as tfh:
            dev_ja = tfh.extractfile("./flores200_dataset/dev/jpn_Jpan.dev")
            dev_en = tfh.extractfile("./flores200_dataset/dev/eng_Latn.dev")
            devtest_ja = tfh.extractfile("./flores200_dataset/devtest/jpn_Jpan.devtest")
            devtest_en = tfh.extractfile("./flores200_dataset/devtest/eng_Latn.devtest")
            
            # dev_meta = tfh.extractfile("./flores200_dataset/metadata_dev.tsv")
            # dev_meta = pd.read_csv(dev_meta, sep="\t")
            # devtest_meta = tfh.extractfile("./flores200_dataset/metadata_devtest.tsv")
            # devtest_meta = pd.read_csv(devtest_meta, sep="\t")
            
            dev_list = []
            for ja_l, en_l in zip(dev_ja.readlines(), dev_en.readlines()):
                dev_list.append({"ja": ja_l.decode("utf-8")[:-1], "en": en_l.decode("utf-8")[:-1]})
            
            devtest_list = []
            for ja_l, en_l in zip(devtest_ja.readlines(), devtest_en.readlines()):
                devtest_list.append({"ja": ja_l.decode("utf-8")[:-1], "en": en_l.decode("utf-8")[:-1]})

        test = Dataset.from_list(dev_list).rename_columns({"en": "en_sentence", "ja": "ja_sentence"})
        test.to_csv(output_path1)
        
        devtest = Dataset.from_list(devtest_list).rename_columns({"en": "en_sentence", "ja": "ja_sentence"})
        devtest.to_csv(output_path2)
        return

    @staticmethod
    def info():
        print(Flores.INFO)
        return
    
    @staticmethod
    def load(which="devtest"):
        assert which in ["dev", "devtest"], "Invalid sub-dataset"
        outname = Flores.OUT_NAMES[0] if which == "dev" else Flores.OUT_NAMES[1]
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{outname}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=outname))
            Flores.create_csv()
            
        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data


    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{EnJaDataset.DATASET_RAW_DIR}/flores"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = f"{output_dir}/flores200_dataset.tar.gz"
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
                url=Flores.DOWNLOAD_URL,
                filename=output_path,
                reporthook=log_progress,
            )
        return
