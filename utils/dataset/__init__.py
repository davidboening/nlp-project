from .jest import JESTDataset
from .massive_translation import MassiveTranslationDataset
from .snow_simplified import SnowSimplifiedDataset
from .tatoeba import TatoebaDataset
from .wiki_corpus import WikiCorpusDataset

__all__ = [
    "JESTDataset", 
    "MassiveTranslationDataset",
    "SnowSimplifiedDataset",
    "TatoebaDataset",
    "WikiCorpusDataset"
]