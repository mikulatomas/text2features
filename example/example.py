import text2features.handlers
import text2features.extractors
import glob

# Text files as input
# Source of the dataset http://mlg.ucd.ie/datasets/bbc.html
filelist = glob.glob('datasets/bbc/*/*.txt')

# Additional stopwords
stopwords = ['mr', 'mrs', 'Mr', 'Mrs', '-which',
             'z.', 'mr.', 'Mr.', 'mrs.', 'Mrs.']

# Extractor for extracting keywords from text
extractor = text2features.extractors.TextRank(
    stopwords=stopwords,  # Additional stopwords
    ignore_words_len=[1],  # Ignore words with length
    candidate_pos=['NOUN', 'PROPN'],  # Which words to use
    window_size=4,  # Window size, used in Text Rank
    min_score=1.8,  # TextRank minimal score to be included
    min_number=5,  # Minimal number of keywords per file
    max_number=10,  # Maximal number of keywords per file
)

# Handler for extracting text from files
handler = text2features.handlers.FileHandler(extractor)

# Process files and save keywords to csv file, build_universum means that output will include file with set of all keywords
handler.process_to_file(filelist, 'example_output.csv', build_universum=True)
