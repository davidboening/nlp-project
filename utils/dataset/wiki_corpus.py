# python libraries
from typing import Tuple, List
import warnings, os, re
from urllib.request import urlretrieve
from zipfile import ZipFile
from xml.etree.ElementTree import ParseError, parse as xml_parse

# external libraries
from datasets import load_dataset
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# local libraries
from .dataset_loader import DatasetLoader


class WikiCorpusDataset:
    OUT_NAME = r"wiki_corpus.csv"
    DOWNLOAD_URL = (
        r"https://github.com/venali/BilingualCorpus/archive/refs/heads/master.zip"
    )
    INFO = (
        "Webpage : https://github.com/venali/BilingualCorpus/\n"
        "Summary : a large scale corpus of manually translated Japanese sentences\n"
        "          extracted from Wikipedia's Kyoto Articles (~500k sentences)"
    )

    @staticmethod
    def create_csv(force_override=False):
        # check processed file presence
        output_path = f"{DatasetLoader.DATASET_PROCESSED_DIR}/{WikiCorpusDataset.OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                DatasetLoader.SKIPPED_MSG_FORMAT.format(
                    file=WikiCorpusDataset.OUT_NAME
                )
            )
            return
        WikiCorpusDataset._download_raw()
        if not os.path.exists(DatasetLoader.DATASET_PROCESSED_DIR):
            os.makedirs(DatasetLoader.DATASET_PROCESSED_DIR)
        # create csv file
        is_xml = re.compile(r"BilingualCorpus-master/wiki_corpus_2.01/[A-Z]{3}/.*\.xml")
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetLoader.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            with ZipFile(
                f"{DatasetLoader.DATASET_RAW_DIR}/wiki_corpus/master.zip", mode="r"
            ) as zhf:
                file_list = zhf.namelist()
                progress_bar = tqdm(desc="Parsing XML files", unit=" Files")
                for file in file_list:
                    if is_xml.match(file):
                        with zhf.open(file, "r") as xml_fh:
                            # if xml parse does not fail add title and sentences
                            if (
                                res := WikiCorpusDataset._parse_wiki_corpus_xml(xml_fh)
                            ) is not None:
                                (ja_t, en_t), sentences, _ = res
                                if en_t is not None and len(en_t) > 1:
                                    en_t = en_t.replace('"', '""')
                                    ja_t = ja_t.replace('"', "")
                                    out_line = f'"{en_t}","{ja_t}"\n'
                                    csv_file.write(out_line.encode("utf-8"))
                                for ja_s, en_s in sentences:
                                    if en_s is None or len(en_s) < 2:
                                        continue
                                    en_s = en_s.replace('"', '""')
                                    ja_s = ja_s.replace('"', "")
                                    out_line = f'"{en_s}","{ja_s}"\n'
                                    csv_file.write(out_line.encode("utf-8"))
                        progress_bar.update(1)
        return

    @staticmethod
    def info():
        print(WikiCorpusDataset.INFO)
        return
    
    @staticmethod
    def stats(en_tokenizer, ja_tokenizer, num_proc=4):
        csv_path = (
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{WikiCorpusDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=WikiCorpusDataset.OUT_NAME))
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
            f"{DatasetLoader.DATASET_PROCESSED_DIR}/{WikiCorpusDataset.OUT_NAME}"
        )
        if not os.path.exists(csv_path):
            print(DatasetLoader.MISSING_FILE_FORMAT.format(file=WikiCorpusDataset.OUT_NAME))
            return
        return load_dataset("csv", data_files=csv_path, **kwargs)

    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{DatasetLoader.DATASET_RAW_DIR}/wiki_corpus"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = f"{output_dir}/master.zip"
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
                url=WikiCorpusDataset.DOWNLOAD_URL,
                filename=output_path,
                reporthook=log_progress,
            )
        return

    @staticmethod
    def _parse_wiki_corpus_xml(
        xml_file_path, debug=False
    ) -> Tuple[Tuple[str, str], List[Tuple[str, str]], str]:
        """Given a path to an xml wiki_corpus file returns title, corpus and id.

        Parameters
        ----------
        xml_file : str
            xml file to parse
        debug : bool
            if True, prints parsing erros when they happen, default is False

        Returns
        -------
        Tuple[Tuple[str, str], List[Tuple[str, str]], str]
            ((ja_title, en_title), [(ja_sentence_1, en_sentence_1),...], id)
        """
        try:
            tree = xml_parse(xml_file_path)
            root = tree.getroot()
            assert root.get("orl") == "ja" and root.get("trl") == "en"
            file_id = root.find("inf").text
            title_data = root.find("tit")
            title_ja = title_data.find("j").text
            title_en_data = title_data.findall("e")[-1]
            assert title_en_data.attrib["type"] == "check"
            title_en = title_en_data.text

            sentences = root.findall(".//*[@id]")
            parallel_corpus = []
            for s in sentences:
                # sec (section) and tit (title) are excluded
                if s.tag in ["par", "sec", "tit"]:
                    continue
                assert s.tag == "sen", f"{s.tag}"
                source_s = s.find("j").text
                target_s_data = s.findall("e")[-1]
                assert target_s_data.attrib["type"] == "check"
                target_s = target_s_data.text
                if source_s is None or target_s is None:
                    continue
                parallel_corpus.append((source_s, target_s))
            return (title_ja, title_en), parallel_corpus, file_id

        except ParseError as e:
            if debug:
                print(e)
            return None
