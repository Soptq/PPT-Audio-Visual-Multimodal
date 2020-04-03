import io, math
import os.path as op


class Segmenter(object):
    ALPHABET = set('abcdefghijklmnopqrstuvwxyz0123456789')
    UNIGRAMS_FILENAME = op.join(
        op.dirname(op.realpath(__file__)),
        'unigrams.txt',
    )
    BIGRAMS_FILENAME = op.join(
        op.dirname(op.realpath(__file__)),
        'bigrams.txt',
    )
    TOTAL = 1024908267229.0
    LIMIT = 500
    WORDS_FILENAME = op.join(
        op.dirname(op.realpath(__file__)),
        'words.txt',
    )

    def __init__(self):
        self.unigrams = {}
        self.bigrams = {}
        self.total = 0.0
        self.limit = 0
        self.words = []

    def load(self):
        self.unigrams.update(self.parse(self.UNIGRAMS_FILENAME))
        self.bigrams.update(self.parse(self.BIGRAMS_FILENAME))
        self.total = self.TOTAL
        self.limit = self.LIMIT
        with io.open(self.WORDS_FILENAME, encoding='utf-8') as reader:
            text = reader.read()
            self.words.extend(text.splitlines())

    @staticmethod
    def parse(filename):
        "Read `filename` and parse tab-separated file of word and count pairs."
        with io.open(filename, encoding='utf-8') as reader:
            lines = (line.split('\t') for line in reader)
            return dict((word, float(number)) for word, number in lines)

    def score(self, word, previous=None):
        "Score `word` in the context of `previous` word."
        unigrams = self.unigrams
        bigrams = self.bigrams
        total = self.total

        if previous is None:
            if word in unigrams:
                # Probability of the given word.

                return unigrams[word] / total

            # Penalize words not found in the unigrams according
            # to their length, a crucial heuristic.

            return 10.0 / (total * 10 ** len(word))

        bigram = '{0} {1}'.format(previous, word)

        if bigram in bigrams and previous in unigrams:
            # Conditional probability of the word given the previous
            # word. The technical name is *stupid backoff* and it's
            # not a probability distribution but it works well in
            # practice.

            return bigrams[bigram] / total / self.score(previous)

        # Fall back to using the unigram probability.

        return self.score(word)

    def isegment(self, text):
        "Return iterator of words that is the best segmenation of `text`."
        memo = dict()

        def search(text, previous='<s>'):
            "Return max of candidates matching `text` given `previous` word."
            if text == '':
                return 0.0, []

            def candidates():
                "Generator of (score, words) pairs for all divisions of text."
                for prefix, suffix in self.divide(text):
                    prefix_score = math.log10(self.score(prefix, previous))

                    pair = (suffix, prefix)
                    if pair not in memo:
                        memo[pair] = search(suffix, prefix)
                    suffix_score, suffix_words = memo[pair]

                    yield (prefix_score + suffix_score, [prefix] + suffix_words)

            return max(candidates())

        # Avoid recursion limit issues by dividing text into chunks, segmenting
        # those chunks and combining the results together. Chunks may divide
        # words in the middle so prefix chunks with the last five words of the
        # previous result.

        clean_text = self.clean(text)
        size = 250
        prefix = ''

        for offset in range(0, len(clean_text), size):
            chunk = clean_text[offset:(offset + size)]
            _, chunk_words = search(prefix + chunk)
            prefix = ''.join(chunk_words[-5:])
            del chunk_words[-5:]
            for word in chunk_words:
                yield word

        _, prefix_words = search(prefix)

        for word in prefix_words:
            yield word

    def segment(self, text):
        "Return list of words that is the best segmenation of `text`."
        return list(self.isegment(text))

    def divide(self, text):
        "Yield `(prefix, suffix)` pairs from `text`."
        for pos in range(1, min(len(text), self.limit) + 1):
            yield (text[:pos], text[pos:])

    @classmethod
    def clean(cls, text):
        "Return `text` lower-cased with non-alphanumeric characters removed."
        alphabet = cls.ALPHABET
        text_lower = text.lower()
        letters = (letter for letter in text_lower if letter in alphabet)
        return ''.join(letters)


alphabet = set('abcdefghijklmnopqrstuvwxyz')
number = set('0123456789')

_segmenter = Segmenter()
clean = _segmenter.clean
load = _segmenter.load
segment = _segmenter.segment


load()

def correct(words):
    word_list = []
    words = words.replace(" ", "").lower()
    is_en = -1
    tem_str = ""
    for word in words:
        if is_en == -1:
            if word in alphabet:
                is_en = 0
            else:
                is_en = 1
            tem_str = word
            continue
        if word in alphabet:
            if is_en == 1: # not en, switch
                word_list.append([tem_str, is_en])
                is_en = 0
                tem_str = word
            else: # not en, not switch
                if word in number:
                    tem_str += " {} ".format(word)
                else:
                    tem_str += word
        else:
            if is_en == 0: # en, switch
                word_list.append([tem_str, is_en])
                is_en = 1
                tem_str = word
            else: # en, not switch
                tem_str += word
    word_list.append([tem_str, is_en])

    for index in range(len(word_list)):
        if word_list[index][1] == 0:
            word_list[index][0] = ' '.join(segment(word_list[index][0]))

    output = ""
    for pair in word_list:
        if pair[1] == 0: # en
            output += " {} ".format(pair[0])
        else:
            output += pair[0]
    return output

if __name__ == "__main__":
    print(correct("hellomynameiseric"))