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


class WMTvat(EnJaDataset):
    DOWNLOAD_URL = {
        "en-ja": {
            "src": r"https://raw.githubusercontent.com/NLP2CT/Variance-Aware-MT-Test-Sets/main/VAT_data/wmt20/vat_newstest2020-enja-src.en.txt",
            "ref": r"https://raw.githubusercontent.com/NLP2CT/Variance-Aware-MT-Test-Sets/main/VAT_data/wmt20/vat_newstest2020-enja-ref.ja.txt"
        },
        "ja-en":{
            "src": r"https://raw.githubusercontent.com/NLP2CT/Variance-Aware-MT-Test-Sets/main/VAT_data/wmt20/vat_newstest2020-jaen-src.ja.txt",
            "ref": r"https://raw.githubusercontent.com/NLP2CT/Variance-Aware-MT-Test-Sets/main/VAT_data/wmt20/vat_newstest2020-jaen-ref.en.txt"
        }
    }
    OUT_NAMES = (r"wmt_vat.en.ja.csv", r"wmt_vat.ja.en.csv")
    INFO = (
        "Webpage    : https://huggingface.co/datasets/gsarti/wmt_vat\n"
        "Paper      : https://openreview.net/forum?id=hhKA5k0oVy5"
        "Summary    : A filtered version of WMT dataset increasing correlation with human\n"
        "             judgement. Contains ja-en, en-ja professional translations for evaluation tasks"
    )

    @staticmethod
    def create_csv(force_override=False):
        # check processed file presence
        output_path_enja = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{WMTvat.OUT_NAMES[0]}"
        )
        output_path_jaen = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{WMTvat.OUT_NAMES[1]}"
        )
        if not force_override and os.path.exists(output_path_jaen) and os.path.exists(output_path_enja):
            print(
                EnJaDataset.SKIPPED_MSG_FORMAT.format(
                    file=WMTvat.OUT_NAMES
                )
            )
            return
        
        WMTvat._download_raw()
        
        if not os.path.exists(EnJaDataset.DATASET_PROCESSED_DIR):
            os.makedirs(EnJaDataset.DATASET_PROCESSED_DIR)
        
        csv_dir = f"{EnJaDataset.DATASET_RAW_DIR}/WMT_vat"
        opt = dict(mode="r", encoding="utf-8")
        # create csv file
        enja_list = []
        with open(f"{csv_dir}/enja-src.en.txt", **opt) as src, open(f"{csv_dir}/enja-ref.ja.txt", **opt) as ref:
            for en_l, ja_l in zip(src.readlines(), ref.readlines()):
                enja_list.append({"ja": ja_l[:-1], "en": en_l[:-1]})
        jaen_list = []
        with open(f"{csv_dir}/jaen-src.ja.txt", **opt) as src, open(f"{csv_dir}/jaen-ref.en.txt", **opt) as ref:
            for ja_l, en_l in zip(src.readlines(), ref.readlines()):
                jaen_list.append({"ja": ja_l[:-1], "en": en_l[:-1]})

        enja = Dataset.from_list(enja_list).rename_columns({"en": "en_sentence", "ja": "ja_sentence"})
        enja.to_csv(output_path_enja)
        
        jaen = Dataset.from_list(jaen_list).rename_columns({"en": "en_sentence", "ja": "ja_sentence"})
        jaen.to_csv(output_path_jaen)
        return

    @staticmethod
    def info():
        print(WMTvat.INFO)
        return
    
    @staticmethod
    def load(which):
        assert which in ["en-ja", "ja-en"], "Invalid sub-dataset"
        outname = WMTvat.OUT_NAMES[0] if which == "ja-en" else WMTvat.OUT_NAMES[1]
        csv_path = (
            f"{EnJaDataset.DATASET_PROCESSED_DIR}/{outname}"
        )
        if not os.path.exists(csv_path):
            print(EnJaDataset.MISSING_FILE_FORMAT.format(file=outname))
            WMTvat.create_csv()
            
        disable_progress_bar()
        data = load_dataset("csv", data_files=csv_path, split="train")
        enable_progress_bar()
        
        return data


    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{EnJaDataset.DATASET_RAW_DIR}/WMT_vat"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        downloads = {
            "enja-src.en.txt" : WMTvat.DOWNLOAD_URL["en-ja"]["src"],
            "enja-ref.ja.txt" : WMTvat.DOWNLOAD_URL["en-ja"]["ref"],
            "jaen-src.ja.txt" : WMTvat.DOWNLOAD_URL["ja-en"]["ref"],
            "jaen-ref.en.txt" : WMTvat.DOWNLOAD_URL["ja-en"]["src"]
        }
        
        outff = output_dir + "/{name}"
        if (force_download or not all(os.path.exists(f"{output_dir}/{f}") for f in downloads.keys())):
            for fname, url in downloads.items():
                progress_bar = None
                def log_progress(c, s, t):
                    nonlocal progress_bar
                    if progress_bar is None:
                        progress_bar = tqdm(
                            desc=f"Downloading {fname}",
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1000,
                        )
                    progress_bar.update(s)
                outfile_path = f"{output_dir}/{fname}"
                urlretrieve(
                    url=url,
                    filename=outfile_path,
                    reporthook=log_progress,
                )
        return
