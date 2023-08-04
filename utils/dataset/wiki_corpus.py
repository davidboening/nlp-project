# python libraries
import warnings
import os
from typing import Tuple, List
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
        output_path = f"{DatasetConfig.DATASET_PROCESSED_DIR}/{DatasetConfig.WIKI_CORPUS_OUT_NAME}"
        if not force_override and os.path.exists(output_path):
            print(DatasetConfig.SKIPPED_MSG_FORMAT.format(file=DatasetConfig.WIKI_CORPUS_OUT_NAME))
            return
        
        with open(output_path, "wb+") as csv_file:
            header_str = DatasetConfig.CSV_HEADER_STR
            csv_file.write(header_str.encode("utf-8"))
            for category in tqdm(WikiCorpusDataset.get_categories(), desc="Parsing XML files"):
                (titles, sentences, _) = WikiCorpusDataset._parse_all_xml_in_category(
                    f"{DatasetConfig.WIKI_CORPUS_RAW_PATH}/{category}"
                )
                for (ja_t, en_t) in titles:
                    out_line = f'"{en_t}","{ja_t}"\n'
                    csv_file.write(out_line.encode("utf-8"))
                for (ja_s, en_s) in sentences:
                    out_line = f'"{en_s}","{ja_s}"\n'
                    csv_file.write(out_line.encode("utf-8"))
        return
    
    
    
    @staticmethod
    def info():
        print(DatasetConfig.WIKI_CORPUS_INFO)
    
    @staticmethod
    def _parse_all_xml_in_category(category_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """Given a path to a category of the wiki_corpus dataset returns the 
        combined corpus of all files in that category.

        Parameters
        ----------
        cat_path : str
            path to a wiki_corpus category

        Returns
        -------
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]
            [(ja_title_1, en_title_1), ...], [(ja_sentence_1, en_sentence_1), ...], [id_1, ...])
        """
        for (_, _, filenames) in os.walk(category_path):
            # extend filenames only
            parallel_corpus, titles, ids = [], [], []
            for file in filenames:
                data = WikiCorpusDataset._parse_wiki_corpus_xml(f"{category_path}/{file}")
                if data is not None:
                    title_both, corpora, id = data
                    titles.append(title_both)
                    parallel_corpus.extend(corpora)
                    ids.append(id)
            # first level only
            break
        return titles, parallel_corpus, ids
    
    @staticmethod
    def _parse_wiki_corpus_xml(xml_file_path:str, debug=False) -> Tuple[Tuple[str, str], List[Tuple[str, str]], str]:
        """Given a path to an xml wiki_corpus file returns title, corpus and id.
        
        Parameters
        ----------
        xml_file : str
            xml file to parse

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
            assert (title_en_data.attrib["type"] == "check")
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
                assert (target_s_data.attrib["type"] == "check")
                target_s = target_s_data.text
                parallel_corpus.append((source_s, target_s))
            return (title_ja, title_en), parallel_corpus, file_id

        except ParseError as e:
            if debug:
                print(e)
            return None

    @staticmethod
    def get_categories():
        """Returns all valid categories for the wiki_corpus."""
        for (_, dirnames, _) in os.walk(DatasetConfig.WIKI_CORPUS_RAW_PATH):
            break
        return dirnames
