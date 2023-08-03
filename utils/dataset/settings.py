# listed datasets are all very high quality
class DatasetConfig:
    """Class containing dataset information"""
    
    # changing order (en,jp) will not affect output columns
    # also spaces are better avoided as they will remain when loading datasets
    CSV_HEADER_STR = "en_sentence,ja_sentence\n" 
    SKIPPED_MSG_FORMAT = "skipped: {file} file already exists!"
    
    HF_DATASET_RAW_DIR = r"./data"
    PROCESSED_DATA_DIR = r"./data-post"

    JESC_RAW_PATH = r"./data/JESC/raw.tar.gz"
    JESC_OUT_NAME = r"jesc.csv"
    JESC_INFO = "Webpage: https://nlp.stanford.edu/projects/jesc/\n" \
                "Paper  : https://arxiv.org/abs/1710.10639\n" \
                "Summary: Japanese-English Subtitle Corpus (2.8M sentences)"

    MASSIVE_TRANSLATION_OUT_NAME = r"massive_translation.csv"
    MASSIVE_TRANSLATION_INFO = "Webpage: https://huggingface.co/datasets/Amani27/massive_translation_dataset\n" \
                               "Summary: dataset derived from AmazonScience/MASSIVE for translation\n" \
                               "         (16k sentences in 10 languages)"

    SNOW_SIMPLIFIED_OUT_NAME = r"snow_simplified.csv"
    SNOW_SIMPLIFIED_INFO = "Webpage: https://huggingface.co/datasets/snow_simplified_japanese_corpus\n" \
                           "Summary: Japanese-English sentence pairs, all Japanese sentences have\n" \
                           "         a simplified counterpart (85k(x2) sentences)"
    
    TATOEBA_OUT_NAME = r"tatoeba.csv"
    TATOEBA_INFO = "Webpage    : https://opus.nlpl.eu/Tatoeba.php\n" \
                   "Webpage(HF): https://huggingface.co/datasets/tatoeba\n" \
                   "Summary    : a collection of sentences from https://tatoeba.org/en/, contains\n" \
                   "             over 400 languages ([en-ja] 200k sentences)"
    
    WIKI_CORPUS_RAW_PATH = r"./data/wiki_corpus_2.01"
    WIKI_CORPUS_OUT_NAME = r"wiki_corpus.csv"
    WIKI_CORPUS_INFO = "Webpage : https://github.com/venali/BilingualCorpus/\n" \
                       "Summary : a large scale corpus of manually translated Japanese sentences\n" \
                       "          extracted from Wikipedia's Kyoto Articles (~500k sentences)"