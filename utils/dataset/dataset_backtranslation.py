# python libraries
from typing import Callable
import os, re, warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# external libraries
import pandas as pd
from datasets import Dataset
from transformers import Seq2SeqTrainer

class EnJaBackTranslation:    
    def create_mBART_backtranslation(trainer : Seq2SeqTrainer, data: Dataset, src_lang: str, tokenizer: Callable, *, chunk_size=1000, gen_config : dict = {}, out_dir="./data-bt", out_name="bt.csv", resume=True):
        """Creates backtranslation from an *ordered* (ideally by lenght) dataset.
        Can resume from previous iteration in case of interruptions.

        Parameters
        ----------
        trainer : Seq2SeqTrainer
            a huggingface seq2seq trainer
        data : Dataset
            a huggingface dataset, containing ["length", "id", "source", "input_ids", "attention_mask"].
            *Should be sorted by "lenght, id" to ensure correct resumption and maximum performance*.
        src_lang : str
            the source language, either "en" or "ja"
        tokenizer : Callable
            the mBART tokenizer
        chunk_size : int, optional
            chucks to compute before saving to disk, by default 1000
        gen_config : dict, optional
            generation config for the trainer, by default {}
        out_dir : str, optional
            file output directory (both for chunks and file), by default "."
        out_name : str, optional
            final csv file name (should end with '.csv'), by default "bt.csv"
        resume : bool, optional
            whenever to resume from previous chuck or not, by default True
        """
        assert src_lang in ["en", "ja"], "Invalid language : should be 'en' or 'ja'"
        trg_lang = "en" if src_lang == "ja" else "ja"
        assert isinstance(trainer, Seq2SeqTrainer), "Invalid trainer passed!"
        assert isinstance(data, Dataset), "Invalid data passed!"
        assert tokenizer is not None and hasattr(tokenizer, "__call__"), "Object passed is not a valid tokenizer!"
        assert chunk_size > 0, "Invalid chunk size passed!"
        assert out_name.endswith(".csv"), "Invalid file name!"
        if os.path.exists(f"{out_dir}/{out_name}"):
            print(f"dataset [{out_dir}/{out_name}] already exists!")
            return
        
        # create directories if missing
        chunk_dir = f"{out_dir}/{out_name[:-4]}"
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir)
        # resume from last chunk, we restart from the last
        # to avoid bad writes in the last interrupted interation
        last_chunk = 0
        if resume:
            # check existing chunks to obtain last one
            pattern = re.compile(r"chunk.(\d+).csv")
            chunk_names = os.listdir(chunk_dir)
            chunk_nums = []
            for cname in chunk_names:
                chunk_nums.append(int(re.match(pattern, cname).groups()[0]))
            last_chunk = max(chunk_nums) if last_chunk > 0 else 0

            # check existing chunks to obtain last to ensure all previous ones exist
            for i in range(last_chunk):
                assert os.path.exists(f"{chunk_dir}/chunk.{i}.csv"), f"chunk #{i} is missing"
            print(f"Resuming from chunck #{last_chunk}")
            del chunk_names, chunk_nums
        # create backtranslation in chunks
        offset, total = last_chunk*chunk_size, len(data)
        nchunks, rem = divmod(total, chunk_size)
        nchunks += (1 if rem > 0 else 0)
        pbar = tqdm(desc="Generating Dataset", unit="chunks", unit_scale=True, unit_divisor=1000, total=nchunks)
        while offset < total:
            # backtranslation target (== source sentence)
            gen_chunk = data.select(range(offset,offset+chunk_size))
            
            # backtranslation source (== model generation)
            gen_out = trainer.predict(gen_chunk, **gen_config)
            gen_chunk_target = tokenizer.batch_decode(gen_out.predictions, skip_special_tokens=True)

            # create new chuck and append to existing csv
            new_chuck = pd.DataFrame({
                f"{src_lang}_sentence" : gen_chunk["source"],
                f"{trg_lang}_sentence" : gen_chunk_target
            })
            new_chuck.to_csv(f"{chunk_dir}/chunk.{last_chunk}.csv", index=False)
            offset += chunk_size
            last_chunk += 1
            pbar.update(1)
        # merge chunks into a single csv file
        for i in range(nchunks):
            df = pd.read_csv(f"{chunk_dir}/chunk.{i}.csv")
            if i == 0:
                df.to_csv(f"{out_dir}/{out_name}", header=True, index=False)
            df.to_csv(f"{out_dir}/{out_name}", mode="a", header=False, index=False)
        return