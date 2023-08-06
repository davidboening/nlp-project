# python libraries
from typing import Tuple, List
import warnings, os, re
from urllib.request import urlretrieve
from zipfile import ZipFile
from xml.etree.ElementTree import ParseError, parse as xml_parse

# external libraries
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm

# local libraries
from .settings import DatasetConfig


class WikiCorpusDataset:
    @staticmethod
    def create_csv(force_override=False):
        # check processed file presence
        output_path = f"{DatasetConfig.DATASET_PROCESSED_DIR}/{DatasetConfig.WIKI_CORPUS_OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(
                DatasetConfig.SKIPPED_MSG_FORMAT.format(
                    file=DatasetConfig.WIKI_CORPUS_OUT_NAME
                )
            )
            return
        WikiCorpusDataset._download_raw()
        if not os.path.exists(DatasetConfig.DATASET_PROCESSED_DIR):
            os.makedirs(DatasetConfig.DATASET_PROCESSED_DIR)
        # create csv file
        is_xml = re.compile(r"BilingualCorpus-master/wiki_corpus_2.01/[A-Z]{3}/.*\.xml")
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            with ZipFile(
                f"{DatasetConfig.DATASET_RAW_DIR}/wiki_corpus/master.zip", mode="r"
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
                                out_line = f'"{en_t}","{ja_t}"\n'
                                csv_file.write(out_line.encode("utf-8"))
                                for ja_s, en_s in sentences:
                                    out_line = f'"{en_s}","{ja_s}"\n'
                                    csv_file.write(out_line.encode("utf-8"))
                        progress_bar.update(1)
        return

    @staticmethod
    def info():
        print(DatasetConfig.WIKI_CORPUS_INFO)

    @staticmethod
    def _download_raw(force_download=False):
        # check raw file presence
        output_dir = f"{DatasetConfig.DATASET_RAW_DIR}/wiki_corpus"
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
                url=DatasetConfig.WIKI_CORPUS_DOWNLOAD_URL,
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
                parallel_corpus.append((source_s, target_s))
            return (title_ja, title_en), parallel_corpus, file_id

        except ParseError as e:
            if debug:
                print(e)
            return None
