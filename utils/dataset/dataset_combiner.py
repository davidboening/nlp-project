# python libraries
from typing import List, Tuple, Callable, NewType
from dataclasses import dataclass
import os, random

# external libraries
from datasets import concatenate_datasets, load_from_disk, Dataset

# local libraries
from .dataset_base import EnJaDataset

DatasetID = NewType('DatasetID', str)

@dataclass
class EnJaDatasetSample:
    """Defines dataset splits

    Parameters
    ----------
    dataset : EnJaDataset
        an EnJaDataset subclass defining a dataset
    nsample : int
        #samples to take from this dataset
    ntokens : Tuple[int, int]
        min and max #tokens allowed per sentence
    """
    dataset : EnJaDataset
    nsample : int
    ntokens : Tuple[int, int]
    


class EnJaDatasetMaker():
    """Prepares a dataset for training from various sources."""
    
    @staticmethod
    def load_dataset(dataset_id : DatasetID) -> Dataset:
        """Returns the dataset with given id if it exists."""
        save_dir = (
            f"{EnJaDataset.DATASET_FINAL_DIR}/{dataset_id}"
        )
        if os.path.exists(save_dir):
            return load_from_disk(save_dir)
        else:
            raise ValueError(EnJaDataset.LOAD_INVALID_ID_FORMAT.format(id=dataset_id))
    
    @staticmethod
    def prepare_dataset(
        dataset_id  : DatasetID,
        dataset_splits: List[EnJaDatasetSample],
        *,
        source_language : str = None,
        model_type : str = None,
        tokenizer : Callable = None,
        encoder_tokenizer : Callable = None,
        decoder_tokenizer : Callable = None,
        num_proc : int = 4,
        seed : int = 42,
    ) -> Dataset:
        """Create a new dataset with given specifics. Or if it exists
        loads it from cache.

        Parameters
        ----------
        dataset_id : DatasetID
            unique identifier for this dataset
        dataset_splits : List[EnJaDatasetSplit]
            a list of EnJaDatasetSplit specifiying which datasets should be combined 
            and how they should be combined. Refer to `EnJaDatasetSample` for specifics.
        source_language : str
            the source language (either "en" or "ja"), required
        model_type : str
            the model type used (either "BERT-GPT2" or "mBART"), required
        tokenizer : Callable
            a mBART huggingface tokenizer, required if model_type == "mBART"
        encoder_tokenizer : Callable
            a BERT huggingface tokenizer, required if model_type == "BERT-GPT2"
        decoder_tokenizer : Callable
            a GPT2 huggingface tokenizer, required if model_type == "BERT-GPT2"
        num_proc : int
            number of workers for multithreading, by default 4
        seed : int
            seed used for sampling, by default 42
            
        Returns
        -------
        Dataset
            a hugging face dataset
        """
        # argument checking
        assert num_proc > 0, "Invalid number of workers."
        assert source_language in ["en", "ja"], "Invalid language."
        assert model_type in ["BERT-GPT2", "mBART"], "Invalid model type."
        if model_type == "BERT-GPT2":
            assert encoder_tokenizer is not None and hasattr(encoder_tokenizer, "__call__"), "Object passed is not a valid tokenizer!"
            assert decoder_tokenizer is not None and hasattr(decoder_tokenizer, "__call__"), "Object passed is not a valid tokenizer!"
            assert tokenizer is None, "Invalid arguments passed to function call!"
        else:
            assert tokenizer is not None and hasattr(tokenizer, "__call__"), "Object passed is not a valid tokenizer!"
            assert encoder_tokenizer is None and decoder_tokenizer is None, "Invalid arguments passed to function call!"
             
        assert all(isinstance(dss, EnJaDatasetSample) for dss in dataset_splits), "Invalid split object!"
        for i, dss in enumerate(dataset_splits):
            assert issubclass(dss.dataset, EnJaDataset), f"Invalid dataset: dataset_splits[{i}] = {dss.dataset.__name__}"
            assert isinstance(dss.nsample, int) and dss.nsample > 0, f"Invalid number of samples: dataset_splits[{i}] = {dss.nsample}"
            assert isinstance(dss.ntokens, Tuple) and len(dss.ntokens) == 2 and 0 <= dss.ntokens[0] <= dss.ntokens[1], f"Invalid number of samples: dataset_splits[{i}] = {dss.ntokens}"
        
        save_dir = (
            f"{EnJaDataset.DATASET_FINAL_DIR}/{dataset_id}"
        )
        if os.path.exists(save_dir):
            print(EnJaDataset.LOAD_FROM_CACHE_FORMAT.format(id=dataset_id))
            return load_from_disk(save_dir)

        random.seed(seed)
        data_list = []
        for ds_split in dataset_splits:
            data = ds_split.dataset.load()
            
            if source_language == "en":
                data = data.rename_columns({
                    "en_sentence": "source",
                    "ja_sentence": "target"
                })
            else: # source_language == "ja":
                data = data.rename_columns({
                    "ja_sentence": "source",
                    "en_sentence": "target"
                })
            
            if model_type == "BERT-GPT2":
                data = data.map(
                    EnJaDatasetMaker._get_map_compute_BERT_GPT2_tokenization(
                        encoder_tokenizer=encoder_tokenizer, 
                        decoder_tokenizer=decoder_tokenizer
                    ), 
                    num_proc=num_proc,
                )
            else: # model_type == "mBART"
                data = data.map(
                    EnJaDatasetMaker._get_map_compute_mBART_tokenization(
                        tokenizer=tokenizer
                    ), 
                    num_proc=num_proc,
                )
  
            data = data.filter(
                EnJaDatasetMaker._get_filter_is_inside_boundaries(ds_split.ntokens),
                num_proc=num_proc,
            )
            
            try:
                sample_ids = random.sample(range(len(data)), ds_split.nsample)
                print(f"sampling: {ds_split.nsample} out of {len(data)}")
                data = data.select(sample_ids)
            except ValueError:
                # sample is bigger than dataset (use all)
                print(f"sampling: using all data ({len(data)})\n")
            
            data.set_format(type="torch")
            data_list.append(data)
        
        # save dataset to cache
        dataset : Dataset = concatenate_datasets(data_list)
        dataset.shuffle(seed)
        dataset.save_to_disk(save_dir)
        
        return dataset

            
    @staticmethod
    def _get_map_compute_BERT_GPT2_tokenization(*, encoder_tokenizer=None, decoder_tokenizer=None):  
        def compute_tokenization(sample):
            src_tokens = encoder_tokenizer(sample["source"], return_tensors="pt")
            trg_tokens = decoder_tokenizer(sample["target"], return_tensors="pt")
            
            sample["length"]         = src_tokens.input_ids.shape[1]
            sample["input_ids"]      = src_tokens.input_ids.flatten()
            sample["attention_mask"] = src_tokens.attention_mask.flatten()
            sample["labels"]         = trg_tokens.input_ids.flatten()
            
            return sample
        
        return compute_tokenization
    
    @staticmethod
    def _get_map_compute_mBART_tokenization(*, tokenizer=None):
        def compute_tokenization(sample):
            tokens = tokenizer(sample["source"], text_target=sample["target"], return_tensors="pt")
            
            sample["length"] = tokens.input_ids.shape[1]
            sample["input_ids"] = tokens.input_ids.flatten()
            sample["attention_mask"] = tokens.attention_mask.flatten()
            sample["labels"] = tokens.labels.flatten()
            
            return sample
        
        return compute_tokenization

    @staticmethod
    def _get_filter_is_inside_boundaries(splice):
        def is_inside_boundaries(sample):
            return splice[0] <= sample["length"] < splice[1]
        
        return is_inside_boundaries