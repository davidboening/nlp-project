# python libraries
import tarfile
import os
# local libraries
from .settings import DatasetConfig

class JESCDataset:
    @staticmethod
    def create_csv(force_override=False):
        output_path = f"{DatasetConfig.PROCESSED_DATA_DIR}/{DatasetConfig.JESC_OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(DatasetConfig.SKIPPED_MSG_FORMAT.format(file=DatasetConfig.JESC_OUT_NAME))
            return
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            with tarfile.open(DatasetConfig.JESC_RAW_PATH, mode="r") as tfh:
                with tfh.extractfile("raw/raw") as fh:
                    while line := fh.readline():
                        line = line.decode()
                        sep = line.find("\t")
                        en_s, jp_s = line[:sep], line[sep+1:-1]
                        out_line = f'"{en_s}", "{jp_s}"\n'
                        csv_file.write(out_line.encode("utf-8"))
        return
    @staticmethod
    def info():
        print(DatasetConfig.JESC_INFO)