# external libraries
from tqdm.autonotebook import tqdm
# python libraries
import os
from typing import Tuple, List
from xml.etree.ElementTree import ParseError, parse as xml_parse


class WikiCorpus:
    def __init__(self, *, 
        split_level="sen", 
        category="ALL", 
        dataset_dir=r"./data/wiki_corpus_2.01", 
        output_dir=r"./data-post/wiki_corpus_2.01"
    ):
        """Initializes a WikiCorpus loader class.

        Parameters
        ----------
        split_level : str
            splits corpus at this level: sen, splits at individial sentences,
            top, at file/topic level.
        category : str
            if "ALL" all categories, otherwise a valid category to restrict to;
            use WikiCorpus().categories to get a full list.
        dataset_dir : str
            a path to the wiki_corpus dataset directory.
        output_dir : str
            a path to a directory in which output files (e.g. csv files) are saved.
        """
        self.split_level = split_level
        self.category = category
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        self.parse_error_list   = []
        self.parse_error_count  = 0
        self.parse_error_errors = []
        self._categories = None

        assert split_level not in ["top"], "'top' is currently disabled"
        assert split_level in ["sen", "top"], "invalid split level"
        assert category in self.categories or category == "ALL", "invalid category"
        assert os.path.exists(dataset_dir), "dataset directory is invalid or does not exist"
        assert os.path.exists(output_dir),  "output directory is invalid or does not exist"


    @property
    def categories(self):
        """Returns all valid categories for the wiki_corpus."""
        if self._categories is None:
            for (_, dirnames, _) in os.walk(self.dataset_dir):
                break
            self._categories = dirnames
        return self._categories

    def _parse_wiki_corpus_xml(self, xml_file:str) -> Tuple[Tuple[str, str], List[Tuple[str, str]], str]:
        """Given a wiki_corpus file name returns title, corpus and id.
        Corpus can be sentence level, paragraph level or topic (file) level.

        Parameters
        ----------
        xml_file : str
            xml file to parse

        Returns
        -------
        Tuple[Tuple[str, str], List[Tuple[str, str]], str]
            ((title_ja, title_en), [(ja1, en1),...], id)
        """
        try:
            tree = xml_parse(xml_file)
            root = tree.getroot()
            assert root.get("orl") == "ja" and root.get("trl") == "en"
            file_id = root.find("inf").text
            title_data = root.find("tit")
            title_ja = title_data.find("j").text
            title_en_data = title_data.findall("e")[-1]
            assert (title_en_data.attrib["type"] == "check")
            title_en = title_en_data.text

            if self.split_level == "sen":
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
            elif self.split_level == "top":
                assert True
            else: assert True
        except ParseError as e:
            self.parse_error_count += 1
            self.parse_error_list.append(xml_file)
            self.parse_error_errors.append(e)
            return None
    
    def _parse_all_xml_in_category(self, cat_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """Given a path to a category of the wiki_corpus dataset returns the 
        combined corpus of all files in that category. Corpus can be sentence
        level, paragraph level or topic (file) level.

        Parameters
        ----------
        cat_path : str
            path to a wiki_corpus category

        Returns
        -------
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]
            [(title_ja1, title_en1), ...], [(ja1, en1), ...], [id1, ...])
        """
        for (_, _, filenames) in os.walk(cat_path):
            # extend filenames only
            parallel_corpus, titles, ids = [], [], []
            for file in tqdm(filenames, position=1, leave=False):
                data = self._parse_wiki_corpus_xml(f"{cat_path}/{file}")
                if data is not None:
                    title_both, corpora, id = data
                    titles.append(title_both)
                    parallel_corpus.extend(corpora)
                    ids.append(id)
            # first level only
            break
        return titles, parallel_corpus, ids
    
    def _save_corpus_as_csv(self, parallel_corpus: List[Tuple[str, str]], path: str) -> None:
        """Given a parallel corpus and a path. Creates a new csv containing the corpus at that path.

        Parameters
        ----------
        parallel_corpus : List[Tuple[str, str]]
            a parallel corpus
        path : str
            path to csv
        """
        with open(path, "wb+") as csv_file:
            header_str = f'ja_source, en_target\n'
            csv_file.write(header_str.encode("utf-8"))
            for (source, target) in parallel_corpus:
                save_str = f'"{source}", "{target}"\n'
                csv_file.write(save_str.encode("utf-8"))
        return
    
    def _save_tiles_as_csv(self, titles: List[Tuple[str, str]], path: str) -> None:
        """Given a list of titles and a path. Creates a new csv containing all titles at that path.

        Parameters
        ----------
        titles : List[Tuple[str, str]]
            a list of titles in two languages
        path : str
            path to csv
        """
        with open(path, "wb+") as csv_file:
            header_str = f'ja_title, en_title\n'
            csv_file.write(header_str.encode("utf-8"))
            for (title_ja, title_en) in titles:
                save_str = f'"{title_ja}", "{title_en}"\n'
                csv_file.write(save_str.encode("utf-8"))
        return
    
    def _create_category_csv(self, category: str) -> None:
        """Given a category of the wiki_corpus generates the correspoding csv file,
        depending on passed parameters.

        Parameters
        ----------
        category: str
            a category
        """
        # if corpus file at given level exists, skip
        fname = f"{self.output_dir}/{category}-{self.split_level}.csv"
        if os.path.exists(fname):
            print(f"skipping file [{fname}], file already exists!")
            return
        titles, parallel_corpus, ids = self._parse_all_xml_in_category(f"{self.dataset_dir}/{category}")
        # if titles file exists, skip
        if not os.path.exists(f"{self.output_dir}/{category}-titles.csv"):
            self._save_tiles_as_csv(titles, f"{self.output_dir}/{category}-titles.csv")
        self._save_corpus_as_csv(parallel_corpus, fname)
        return

    def create_csv(self):
        """Creates all required wiki_corpus csv files. Does nothing if the files already exist."""
        if self.category == "ALL":
            for category in tqdm(self.categories):
                self._create_category_csv(category)
        else:
            self._create_category_csv(self.category)

    def get_errors(self) -> Tuple[int, List[str], List[ParseError]]:
        """Returns parsing errors.

        Returns
        -------
        Tuple[int, List[str], List[ParseError]]
            count, file_names, errors
        """
        return self.parse_error_count, self.parse_error_list, self.parse_error_errors

