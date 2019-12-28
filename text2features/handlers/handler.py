import os
import csv
import math


class FileHandler():
    """Using given extractor to extract keywords from given list of text files."""

    def __init__(self, extractor):
        self.extractor = extractor

    def process(self, files, include_filename=False):
        """Generator which process list of given text files and returns keywords."""
        for file in files:
            with open(file, 'r', errors='ignore') as f:
                keywords = self.extractor.extract(f.read())

                if include_filename:
                    yield (os.path.basename(file), *keywords)
                else:
                    yield tuple(keywords)

    def process_to_file(self,
                        files,
                        output,
                        delimiter=',',
                        lineterminator='\r\n',
                        build_universum=False):
        """Extract all keywords from text files to .csv file"""
        if build_universum:
            universum = set()

        with open(output, 'w') as f:
            writer = csv.writer(f, delimiter=delimiter,
                                lineterminator=lineterminator)

            keywords_with_filename = tuple(
                self.process(files, include_filename=True))
            writer.writerows(keywords_with_filename)

            if build_universum:
                universum.update(*[keywords[1:]
                                   for keywords in keywords_with_filename])

        with open(output.replace('.csv', '_universum.csv'), 'w') as f:
            f.write("\n".join(sorted(universum)))
