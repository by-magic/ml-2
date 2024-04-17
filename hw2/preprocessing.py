from dataclasses import dataclass
from typing import Dict, List, Tuple
from xml.etree.ElementTree import fromstring, ElementTree
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    # with gelp of chatgpt
    with open(filename, "r") as f:
        content = f.read()
        
    content_text = content.replace("&", "*")
 
    tree = ElementTree(fromstring(content_text))
    root = tree.getroot()

    all_sentences = []
    all_targets = []

    get_pairs = lambda x: list(map(lambda x: x.replace("*", "&"), x.split(' '))) if x else []
    get_alignment = lambda x: list(map(lambda x: (int(x.split('-')[0]), int(x.split('-')[1])), x.split(' '))) if x else []
    
    for tmp in root:
        source = get_pairs(tmp[0].text)
        target = get_pairs(tmp[1].text)
        sure = get_alignment(tmp[2].text)
        possible = get_alignment(tmp[3].text)
        
        sentence_pair = SentencePair(source=source, target=target)
        alignment = LabeledAlignment(sure=sure, possible=possible)
        all_sentences.append(sentence_pair)
        all_targets.append(alignment)

    return all_sentences, all_targets



def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    # with gelp of chatgpt
    get_all_words = lambda x: np.unique(np.array([val for sublist in x for val in sublist]))
    
    source_sentences = [x.source for x in sentence_pairs]
    target_sentences = [x.target for x in sentence_pairs]
    
    source_words = get_all_words(source_sentences)
    target_words = get_all_words(target_sentences)

    source_cnt = Counter(np.array([val for sublist in source_sentences for val in sublist]).flatten())
    target_cnt = Counter(np.array([val for sublist in target_sentences for val in sublist]).flatten())

    if freq_cutoff is not None:
        source_words_sorted = [word for word, _ in source_cnt.most_common(freq_cutoff)]
        target_words_sorted = [word for word, _ in target_cnt.most_common(freq_cutoff)]
        
        source_dict = {word: idx for idx, word in enumerate(source_words_sorted)}
        target_dict = {word: idx for idx, word in enumerate(target_words_sorted)}
    else:
        source_dict = {word: idx for idx, word in enumerate(source_words)}
        target_dict = {word: idx for idx, word in enumerate(target_words)}
    
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    # with gelp of chatgpt
    ans = []

    for sentence in sentence_pairs:
        source_indices = np.array([source_dict.get(x, -1) for x in sentence.source])
        target_indices = np.array([target_dict.get(x, -1) for x in sentence.target])
        
        if np.all(source_indices != -1) and np.all(target_indices != -1):
            ans.append(TokenizedSentencePair(source_indices, target_indices))

    return ans

