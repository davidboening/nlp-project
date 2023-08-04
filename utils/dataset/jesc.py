# python libraries
import tarfile, os, warnings
from urllib.request import urlretrieve

# external libraries
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# local libraries
from .settings import DatasetConfig


class JESCDataset:
    @staticmethod
    def create_csv(force_override=False):
        # check processed file presence
        output_path = (
            f"{DatasetConfig.DATASET_PROCESSED_DIR}/{DatasetConfig.JESC_OUT_NAME}"
        )
        if not force_override and os.path.exists(output_path):
            print(
                DatasetConfig.SKIPPED_MSG_FORMAT.format(
                    file=DatasetConfig.JESC_OUT_NAME
                )
            )
            return
        JESCDataset._download_raw()
        # create csv file
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            with tarfile.open(
                f"{DatasetConfig.DATASET_RAW_DIR}/JESC/raw.tar.gz", mode="r"
            ) as tfh:
                with tfh.extractfile("raw/raw") as fh:
                    while line := fh.readline():
                        line = line.decode()
                        sep = line.find("\t")
                        en_s, jp_s = line[:sep], line[sep + 1 : -1]
                        out_line = f'"{en_s}","{jp_s}"\n'
                        csv_file.write(out_line.encode("utf-8"))
        return

    @staticmethod
    def info():
        print(DatasetConfig.JESC_INFO)

    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{DatasetConfig.DATASET_RAW_DIR}/JESC"
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
                url=DatasetConfig.JESC_DOWNLOAD_URL,
                filename=output_path,
                reporthook=log_progress,
            )
        return
