# text2features
Extracts boolean features (keywords) from text. Work in progress, do not use in important projects. Library is able to output keywords to files or variables. This library can be used as input to Formal Concept Analysis (https://en.wikipedia.org/wiki/Formal_concept_analysis).

TextRank extractor based on https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0

## Example

example.py
```python
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

# Process files and save keywords to csv file.
# Parameter 'build_universum' means that output will include file with set of all keywords.
handler.process_to_file(filelist, 'example_output.csv', build_universum=True)
```

example_output.csv
First row is filename, rest are keywords related to input file.
```
entertainment_208.txt,duran,vh1,album,interview,bon
entertainment_234.txt,elvis,chart,number,place,week
entertainment_220.txt,award,interactive,website,tv,category
entertainment_036.txt,hanks,year,square,boy,story
entertainment_022.txt,ballet,dancer,rad,child,dance
entertainment_181.txt,evans,stall,sofa,sale,piece
entertainment_195.txt,aid,relief,world,victim,attack
entertainment_142.txt,new,york,group,bbc,sound,music,act,band,uk,radio
entertainment_156.txt,soul,r&b,awards,nomination,album,music
entertainment_383.txt,film,director,vabres,alan,death
...
```

example_output_universum.csv
One keyword per line.
```
1950
1970
1980
200lr
280bn
3gsm
7e7
a350
aaa
aaas
aaliyah
abba
abbas
abc
abortion
absa
ac
academy
acc
access
accident
account
accounting
acquisition
act
action
activity
actor
actress
ad
...
```


