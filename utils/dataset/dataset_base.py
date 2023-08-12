# python libraries
from abc import ABC, abstractmethod

# external libraries
from datasets import Dataset



# listed datasets are all very high quality
class EnJaDataset(ABC):
    """Template Class for datasets"""

    CSV_HEADER_STR = "en_sentence,ja_sentence\n"
    SKIPPED_MSG_FORMAT = "skipped: {file} file already exists!"
    MISSING_FILE_FORMAT = "warning: {file} not found, creating csv first ..."
    LOAD_FROM_CACHE_FORMAT = 'skipped: loaded dataset with id="{id}" from existing cache.'
    LOAD_INVALID_ID_FORMAT = 'dataset with id="{id}" was not found.'

    DATASET_RAW_DIR = r"./data-raw"
    DATASET_PROCESSED_DIR = r"./data-csv"
    DATASET_FINAL_DIR = r"./data-fin"

    @abstractmethod
    def create_csv():
        """Creates a single csv file of the dataset"""
        pass
    
    @abstractmethod
    def info():
        """Prints a short description of the dataset"""
        pass

    @abstractmethod
    def load(**kwargs) -> Dataset:
        """Returns a DataFrame of the dataset"""
        pass

