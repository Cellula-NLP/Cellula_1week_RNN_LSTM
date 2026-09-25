"""
Preprocessing utilities for multi-label toxic comment classification.

Extracted from the notebook so the same cleaning / tokenization / vocabulary
logic can be reused without re-running the whole notebook.

Usage:
    from preprocessing import clean_text, tokenize, Vocabulary
"""

import re
import html
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer


# --- NLTK data --------------------
for pkg in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


# --- Constants ---------------------------------------------------------------
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MAX_LEN  = 220
MIN_FREQ = 2

_lemmatizer = WordNetLemmatizer()

_LB = r"(?<![a-zA-Z])"   # left boundary: not preceded by a letter

# Deobfuscation patterns — order matters only in that we run all of them.
RE_PATTERNS = {
    "fuck":    [rf"{_LB}f\*+c?k?", rf"{_LB}f[\W_]*u[\W_]*c[\W_]*k", rf"{_LB}fuk\w*", rf"{_LB}fck\w*", rf"{_LB}wtf"],
    "shit":    [rf"{_LB}s[\W_]*h[\W_]*i[\W_]*t", rf"{_LB}\$hit"],
    "bitch":   [rf"{_LB}b[\W_]*i[\W_]*t[\W_]*c[\W_]*h", rf"{_LB}b!tch", rf"{_LB}biatch"],
    "asshole": [rf"{_LB}a[\W_]*s+[\W_]*h[\W_]*o+[\W_]*l+[\W_]*e"],
    "ass":     [rf"{_LB}ass(?![a-zA-Z])", rf"{_LB}@\$\$"],
    "dick":    [rf"{_LB}d[\W_]*i[\W_]*c[\W_]*k"],
    "cunt":    [rf"{_LB}c[\W_]*u[\W_]*n[\W_]*t"],
    "nigger":  [rf"{_LB}n[\W_]*i[\W_]*g+[\W_]*[ae]+[\W_]*r", rf"{_LB}n3gr"],
    "faggot":  [rf"{_LB}f[\W_]*a[\W_]*g+[\W_]*[o0]+[\W_]*t", rf"{_LB}fagot"],
    "retard":  [rf"{_LB}r[\W_]*e[\W_]*t[\W_]*a[\W_]*r[\W_]*d"],
    "idiot":   [rf"{_LB}i[\W_]*d[\W_]*i[\W_]*o[\W_]*t"],
    "dumb":    [rf"{_LB}d[\W_]*u[\W_]*m[\W_]*b"],
    "rape":    [rf"{_LB}r[\W_]*a[\W_]*p[\W_]*e"],
    "kill":    [rf"{_LB}k[\W_]*i[\W_]*l[\W_]*l"],
    "hate":    [rf"{_LB}h[\W_]*a[\W_]*t[\W_]*e"],
    "hitler":  [rf"{_LB}hitler"],
}

_RE_COMPILED = {
    canon: [re.compile(p, re.IGNORECASE) for p in pats]
    for canon, pats in RE_PATTERNS.items()
}

_url_re  = re.compile(r"https?://\S+|www\.\S+")
_ip_re   = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_html_re = re.compile(r"<[^>]+>")
_wiki1   = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")
_wiki2   = re.compile(r"\[\[([^\]]+)\]\]")
_tmpl    = re.compile(r"\{\{.*?\}\}")
_repeat  = re.compile(r"(.)\1{2,}")


# --- Core functions ----------------------------------------------------------
def _deobfuscate(text: str) -> str:
    """Collapse obfuscated profanity ('f*ck', 'f.u.c.k') to canonical tokens."""
    for canon, pats in _RE_COMPILED.items():
        for p in pats:
            text = p.sub(f" {canon} ", text)
    return text


def _lemmatize(text: str) -> str:
    return " ".join(
        _lemmatizer.lemmatize(
            _lemmatizer.lemmatize(_lemmatizer.lemmatize(w, pos="n"), pos="v"), pos="a"
        )
        for w in text.split()
    )


def clean_text(text: str, do_lemma: bool = True) -> str:
    """Full cleaning pipeline. Order matters: deobfuscate BEFORE stripping punctuation."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = _html_re.sub(" ", text)
    text = _wiki1.sub(r"\2", text)
    text = _wiki2.sub(r"\1", text)
    text = _tmpl.sub(" ", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"~{3,}", " ", text)
    text = _url_re.sub(" <url> ", text)
    text = _ip_re.sub(" <ip> ", text)
    text = text.lower()
    text = _deobfuscate(text)
    text = re.sub(r"[^a-z0-9\s'.,!?<>]", " ", text)
    text = _repeat.sub(r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _lemmatize(text) if do_lemma else text


def tokenize(text: str):
    """Regex tokenizer matching the notebook."""
    return re.findall(r"<url>|<ip>|[a-z]+(?:['\-][a-z]+)*|\d+|[!?.,;:]", text)


# --- Vocabulary --------------------------------------------------------------
class Vocabulary:
    """Simple frequency-thresholded vocabulary with pad/unk at indices 0 and 1."""

    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.vocab_size = 2

    def build_vocab(self, texts):
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        for w, n in c.items():
            if n >= self.min_freq:
                self.word2idx[w] = self.vocab_size
                self.vocab_size += 1
        print("Vocab size:", self.vocab_size)

    def numericalize(self, text):
        return [self.word2idx.get(t, 1) for t in tokenize(text)]