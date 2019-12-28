import spacy
import math

DEFAULT_STOPWORDS = []
DEFAULT_SPACY_MODEL_NAME = 'en_core_web_sm'
DEFAULT_CANDIDATE_POS = ['NOUN', 'PROPN', 'VERB']
DEFAULT_IGNORE_WORDS_LEN = [0]
DEFAULT_MIN_NUMBER = 0
DEFAULT_MAX_NUMBER = math.inf


class Extractor():
    """Basic extractor, for inheritance"""

    def __init__(self,
                 stopwords,
                 spacy_model_name,
                 candidate_pos,
                 ignore_words_len,
                 min_number,
                 max_number):

        if stopwords is None:
            stopwords = DEFAULT_STOPWORDS

        if spacy_model_name is None:
            spacy_model_name = DEFAULT_SPACY_MODEL_NAME

        if candidate_pos is None:
            candidate_pos = DEFAULT_CANDIDATE_POS

        if ignore_words_len is None:
            ignore_words_len = DEFAULT_IGNORE_WORDS_LEN

        if min_number is None:
            min_number = DEFAULT_MIN_NUMBER

        if max_number is None:
            max_number = DEFAULT_MAX_NUMBER

        self.nlp = spacy.load(spacy_model_name)
        self.set_stopwords(stopwords)
        self.candidate_pos = candidate_pos
        self.ignore_words_len = ignore_words_len
        self.min_number = min_number
        self.max_number = max_number

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in stopwords:
            self.nlp.vocab[word].is_stop = True
