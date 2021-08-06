import re

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words

from typing import List, Set

# create a set of English words using a corpus
stop_words: List = stopwords.words("english")
vocab: Set = set(words.words()) - set(stop_words)
lemmatizer = WordNetLemmatizer()
words_set: Set = {lemmatizer.lemmatize(word) for word in vocab}


def tokenize(text: str) -> List:
    """
    Tokenize and lemmatize text

    :param text: Text to be tokenized
    :return: List of lemmatized word tokens
    """

    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [
        lemmatizer.lemmatize(token).strip() for token in tokens if token in words_set
    ]

    return clean_tokens
