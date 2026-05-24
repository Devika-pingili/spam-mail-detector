# -*- coding: utf-8 -*-
"""
Text preprocessing utilities for spam detection.
Uses NLTK for tokenization, stopwords, and Porter stemming.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available (safe to call multiple times)
for _resource in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower() if text else ""


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove digits from text."""
    return re.sub(r"\d+", "", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords from token list."""
    return [t for t in tokens if t not in _stop_words]


def tokenize(text: str) -> list[str]:
    """Split text into word tokens."""
    return word_tokenize(text)


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to each token."""
    return [_stemmer.stem(t) for t in tokens]


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple spaces into one and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single message.

    Steps: lowercase -> remove punctuation -> remove numbers ->
           tokenize -> remove stopwords -> stem -> join -> trim spaces

    Example:
        Input:  "Congratulations!!! You won $1000 prize"
        Output: "congratul won prize"
    """
    if not text or not str(text).strip():
        return ""

    text = to_lowercase(str(text))
    text = remove_punctuation(text)
    text = remove_numbers(text)

    tokens = tokenize(text)
    tokens = [t for t in tokens if len(t) > 1]
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)

    result = " ".join(tokens)
    return remove_extra_spaces(result)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Apply preprocessing to a list of texts."""
    return [preprocess_text(t) for t in texts]
