from __future__ import annotations

import re
from typing import List

from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from nltk.corpus import stopwords

SAFE_DIV = 1e-4

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not",
    "can't've": "can not have", "'cause": "because", "could've": "could have",
    "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is", "how'd": "how did",
    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
    "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
    "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not",
    "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have",
    "so's": "so is", "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there would", "there'd've": "there would have",
    "there's": "there is", "they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are", "what's": "what is",
    "what've": "what have", "when's": "when is", "when've": "when have",
    "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is",
    "who've": "who have", "why's": "why is", "why've": "why have",
    "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "you'd": "you would",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have",
}

def preprocess(text: str) -> str:
    if text is None:
        return ""

    q = str(text).lower().strip()
    q = BeautifulSoup(q, "html.parser").get_text(" ")
    q = q.replace("%", " percent ")
    q = q.replace("$", " dollar ")
    q = q.replace("₹", " rupee ")
    q = q.replace("€", " euro ")
    q = q.replace("@", " at ")
    q = q.replace("[math]", " ")

    q = q.replace(",000,000,000", "b")
    q = q.replace(",000,000", "m")
    q = q.replace(",000", "k")
    q = re.sub(r"([0-9]+)000000000\b", r"\1b", q)
    q = re.sub(r"([0-9]+)000000\b", r"\1m", q)
    q = re.sub(r"([0-9]+)000\b", r"\1k", q)

    # Preserve the notebook's decontraction idea while avoiding a costly
    # regex pass over the full contraction dictionary.
    q = re.sub(r"\b[\w']+\b", lambda m: CONTRACTIONS.get(m.group(0), m.group(0)), q)
    q = re.sub(r"[^a-z0-9\s!?.,%$€₹@_-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _token_sets(q1: str, q2: str):
    t1, t2 = q1.split(), q2.split()
    s1, s2 = set(t1), set(t2)
    stop = _get_stopwords()
    nw1, nw2 = s1 - stop, s2 - stop
    sw1, sw2 = s1 & stop, s2 & stop
    return t1, t2, s1, s2, nw1, nw2, sw1, sw2

_STOP_WORDS = None

def _get_stopwords():
    global _STOP_WORDS
    if _STOP_WORDS is None:
        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            # Keeps inference usable when the NLTK corpus is unavailable.
            _STOP_WORDS = {
                "a", "an", "the", "is", "are", "was", "were", "to", "of",
                "in", "on", "for", "and", "or", "what", "where", "when",
                "why", "how", "who", "which", "do", "does", "did", "i",
                "you", "it", "we", "they", "he", "she"
            }
    return _STOP_WORDS

def _safe_ratio(num: float, den: float) -> float:
    return num / (den + SAFE_DIV)

def longest_common_substring_ratio(q1: str, q2: str) -> float:
    # Dynamic programming, optimized for the usually-short question strings.
    if not q1 or not q2:
        return 0.0
    if len(q1) > len(q2):
        q1, q2 = q2, q1
    prev = [0] * (len(q2) + 1)
    best = 0
    for i, c1 in enumerate(q1, 1):
        cur = [0] * (len(q2) + 1)
        for j, c2 in enumerate(q2, 1):
            if c1 == c2:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best / (min(len(q1), len(q2)) + 1)

FEATURE_NAMES = [
    "q1_len", "q2_len", "q1_num_words", "q2_num_words",
    "word_common", "word_total", "word_share",
    "cwc_min", "cwc_max", "csc_min", "csc_max",
    "ctc_min", "ctc_max", "last_word_eq", "first_word_eq",
    "abs_len_diff", "mean_len", "longest_substr_ratio",
    "fuzz_ratio", "fuzz_partial_ratio", "token_sort_ratio", "token_set_ratio"
]

def pair_features(question1: str, question2: str) -> List[float]:
    q1 = preprocess(question1)
    q2 = preprocess(question2)

    t1, t2, s1, s2, nw1, nw2, sw1, sw2 = _token_sets(q1, q2)

    common = len(s1 & s2)
    total = len(s1) + len(s2)
    common_nonstop = len(nw1 & nw2)
    common_stop = len(sw1 & sw2)
    common_tokens = len(s1 & s2)

    return [
        len(q1),
        len(q2),
        len(t1),
        len(t2),
        common,
        total,
        round(common / total, 4) if total else 0.0,
        _safe_ratio(common_nonstop, min(len(nw1), len(nw2))),
        _safe_ratio(common_nonstop, max(len(nw1), len(nw2))),
        _safe_ratio(common_stop, min(len(sw1), len(sw2))),
        _safe_ratio(common_stop, max(len(sw1), len(sw2))),
        _safe_ratio(common_tokens, min(len(t1), len(t2))),
        _safe_ratio(common_tokens, max(len(t1), len(t2))),
        float(bool(t1 and t2 and t1[-1] == t2[-1])),
        float(bool(t1 and t2 and t1[0] == t2[0])),
        abs(len(t1) - len(t2)),
        (len(t1) + len(t2)) / 2.0 if (t1 or t2) else 0.0,
        longest_common_substring_ratio(q1, q2),
        float(fuzz.QRatio(q1, q2)),
        float(fuzz.partial_ratio(q1, q2)),
        float(fuzz.token_sort_ratio(q1, q2)),
        float(fuzz.token_set_ratio(q1, q2)),
    ]

def clean_pair(question1: str, question2: str):
    return preprocess(question1), preprocess(question2)
