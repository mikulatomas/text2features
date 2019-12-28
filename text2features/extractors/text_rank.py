from collections import OrderedDict
from .extractor import Extractor
import numpy as np
import math


class TextRank(Extractor):
    """
    Extract keywords via TextRank from text.
    Modified example based on # https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0
    Original paper about TextRank https://www.aclweb.org/anthology/W04-3252.pdf
    """

    def __init__(self,
                 min_score=1,
                 window_size=4,
                 stopwords=None,
                 spacy_model_name=None,
                 candidate_pos=None,
                 ignore_words_len=None,
                 min_number=None,
                 max_number=None):

        super().__init__(stopwords,
                         spacy_model_name,
                         candidate_pos,
                         ignore_words_len,
                         min_number,
                         max_number)

        # Values taken from article on towardsdatascience
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps

        self.window_size = window_size
        self.min_score = min_score

    def __sentence_segment(self, doc):
        """Filter sentences to extract only word type based on candidate_pos and word length."""
        sentences = []
        for sent in doc.sents:
            selected_words = []

            for token in sent:
                if (token.pos_ in self.candidate_pos
                        and token.is_stop is False
                        and len(token.lemma_) not in self.ignore_words_len):
                    selected_words.append(token.lemma_.lower())

            sentences.append(selected_words)
        return sentences

    def __get_vocab(self, sentences):
        """Build vocab from every word and its possition."""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def __get_token_pairs(self, sentences):
        """Build token_pairs from windows in sentences."""
        token_pairs = set()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + self.window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    token_pairs.add(pair)
        return token_pairs

    def __symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def __get_matrix(self, vocab, token_pairs):
        """Get normalized matrix."""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.__symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        # this is ignore the 0 element in norm
        g_norm = np.divide(g, norm, where=norm != 0)

        return g_norm

    def extract(self, text):
        """Extract all keywords from given text."""

        # Pare text by spaCy
        doc = self.nlp(text)

        # Filter sentences
        sentences = self.__sentence_segment(doc)  # list of list of words

        # Build vocabulary
        vocab = self.__get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.__get_token_pairs(sentences)

        # Get normalized matrix
        g = self.__get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for _ in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        # First try to find keywords, only using keywords filtered by TextRank score
        keywords = tuple(map(lambda x: x[0], filter(
            lambda x: x[1] >= self.min_score, node_weight.items())))

        # If there is not enough keywords, other keywords will be used (based on score)
        if len(keywords) < self.min_number:
            keywords = tuple(map(lambda x: x[0], sorted(
                node_weight.items(), key=lambda x: x[1], reverse=True)))
            keywords = keywords[:self.min_number]
        # If there is to many keywords, they will be filtered.
        elif len(keywords) > self.max_number:
            keywords = keywords[:self.max_number]

        return keywords
