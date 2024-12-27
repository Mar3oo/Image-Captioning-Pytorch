#______________________________________________________
# I tried to use the Vocabulary from MIT one but I think it is not updated 
# and there is a lot of dismatches between the versions of libraries and that 
# raises lots of errors 
# So here is a copy of the class but with some modifications
#______________________________________________________

import json
import nltk
from collections import Counter
from collections.abc import MutableMapping
import csv

class Vocab(MutableMapping):
    def __init__(self, vocab_threshold, vocab_file, start_word="<start>", end_word="<end>", unk_word="<unk>", annotations_file=None, vocab_from_file=True):
        """
        Initialize the vocabulary.
        
        Args:
            vocab_threshold: Minimum frequency for a word to be included in the vocabulary.
            vocab_file: Path to the vocab file.
            start_word: Word to denote the beginning of a sentence.
            end_word: Word to denote the end of a sentence.
            unk_word: Word to denote unknown words.
            annotations_file: Path to a CSV file containing the dataset annotations (for training).
            vocab_from_file: Whether to load the vocabulary from a file or generate a new one.
        """
        self.vocab_threshold = vocab_threshold
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Special tokens
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)

        if vocab_from_file:
            self.load_vocab(vocab_file)
        elif annotations_file:
            self.build_vocab(annotations_file)
            self.save_vocab(vocab_file)

    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx[self.unk_word])
    
    def __setitem__(self, key, value):
        self.word2idx[key] = value

    def __delitem__(self, key):
        del self.word2idx[key]

    def __iter__(self):
        return iter(self.word2idx)

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, annotations_file):
        """Build vocabulary from captions in a CSV file."""
        counter = Counter()
        with open(annotations_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header line ('image', 'caption')
            for row in reader:
                caption = row[1]  # The second column contains the caption
                tokens = nltk.tokenize.word_tokenize(caption.lower())  # Tokenize the caption
                counter.update(tokens)

        # Add words that appear more than the threshold
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
        for word in words:
            self.add_word(word)

    def save_vocab(self, vocab_file):
        """Save vocabulary to a file."""
        with open(vocab_file, 'w') as f:
            json.dump(self.word2idx, f)

    def load_vocab(self, vocab_file):
        """Load vocabulary from a file."""
        with open(vocab_file, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __call__(self, word):
        """Return the index of a word."""
        return self[word]
