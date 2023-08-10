# external libraries
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import numpy as np

# python libraries
from abc import ABC, abstractmethod



# listed datasets are all very high quality
class DatasetLoader(ABC):
    """Class containing dataset information"""

    CSV_HEADER_STR = "en_sentence,ja_sentence\n"
    SKIPPED_MSG_FORMAT = "skipped: {file} file already exists!"
    MISSING_FILE_FORMAT = "failure: {file} not found, call .create_csv() first!"

    DATASET_RAW_DIR = r"./data-raw"
    DATASET_PROCESSED_DIR = r"./data-csv"

    @abstractmethod
    def create_csv():
        """Creates a single csv file of the dataset"""
        pass
    
    @abstractmethod
    def info():
        """Returns a short description of the dataset"""
        pass

    @abstractmethod
    def stats(dataset_path, *, en_tokenizer=None, ja_tokenizer=None, num_proc=4):
        """Returns statistics of the dataset"""
        assert (en_tokenizer is not None and hasattr(en_tokenizer, "__call__")), "Object passed is not a valid tokenizer!"
        assert (ja_tokenizer is not None and hasattr(ja_tokenizer, "__call__")), "Object passed is not a valid tokenizer!"
        
        disable_progress_bar()
        data = load_dataset("csv", data_files=dataset_path)
        enable_progress_bar()

        def _stats(x):
            # load_dataset loads "None" as None in csv files
            if x["en_sentence"] is None: 
                x["en_sentence"] = "None"
            x["en_len"] = len(en_tokenizer(x["en_sentence"]).input_ids)
            x["ja_len"] = len(ja_tokenizer(x["ja_sentence"]).input_ids)
            return x
        
        data_stats = data.map(_stats, num_proc=num_proc)


        en_len = np.array(data_stats["train"]["en_len"])
        ja_len = np.array(data_stats["train"]["ja_len"])

        print(
            f"Showing statistic for {en_len.size:,} sentences:\n"
            f"\ten[tokens] : Avg. {en_len.mean():5.2f} | Min. {en_len.min():5} | Max. {en_len.max():5} |"
            f" >32. {(en_len > 32).sum():7,} | >64. {(en_len > 64).sum():5} |"
            f" >128. {(en_len > 128).sum():5} | >256. {(en_len > 256).sum():5}\n"
            f"\tja[tokens] : Avg. {ja_len.mean():5.2f} | Min. {ja_len.min():5} | Max. {ja_len.max():5} |"
            f" >32. {(ja_len > 32).sum():7,} | >64. {(ja_len > 64).sum():5} |"
            f" >128. {(ja_len > 128).sum():5} | >256. {(ja_len > 256).sum():5}"
        )
