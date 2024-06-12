"""
Tools to pre-process data and prepare it for SVM input.
"""

import collections
import csv
import typing

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag


# Init NLTK
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")


stopwords: list[str] | None = None
lexicon: collections.OrderedDict[str, typing.Any] = collections.OrderedDict()
label_map = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}
replacements = {
    "\\u2019": "'",
    "\\u002c": ",",
}


def clean_text(text: str) -> str:
    """
    Clean text by removing punctuation and making replacements.
    """
    text = text.lower()
    # Replace by correct character
    for s, rep in replacements.items():
        text = text.replace(s, rep)
    # Remove punctuation (BAD IDEA)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def get_wordnet_pos(tag: str) -> str:
    """
    Returns Part of Speech tag representation as defined by WordNet.
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatise_words(words: list[str]) -> list[str]:
    """
    Returns lemmas from list of words, applying WordNetLemmatizer.
    """
    pos_tags = pos_tag(words)
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmas


def remove_stopwords(words: list[str]) -> list[str]:
    """
    Returns the given list of words without stopwords.
    """
    global stopwords
    if not stopwords:
        stopwords = open("res/stopwords-en.txt", "r").read().splitlines()
    return [w for w in words if w not in stopwords]


def extract_words(text: str) -> list[str]:
    """
    Extract useful words from a line of text.
    """
    text = clean_text(text)
    words = word_tokenize(text)
    words = lemmatise_words(words)
    # Removing stopwords negatively impacts accuracy
    # words = remove_stopwords(words)
    return words


def extract_lexicon(in_file: str, out_file: str) -> None:
    """
    Extracts the lexicon from a source file.
    """
    with open(out_file, "w") as f_out:
        with open(in_file, "r") as csv_in:
            reader = csv.reader(csv_in, delimiter='\t', quotechar='"')
            for line in reader:
                _id, emotion, text = line

                # Count words in line and add to lexicon
                words = extract_words(text)
                word_count: dict[str, int] = {}
                for w in words:
                    word_count[w] = word_count.get(w, 0) + 1
                    lexicon[w] = None

                # Generate SVM input line
                svm_line = f"{label_map[emotion]}"
                for idx, w in enumerate(lexicon.keys()):
                    if w in words:
                        svm_line += f" {idx + 1}:{word_count[w]}"
                f_out.write(f"{svm_line}\n")


def main() -> None:
    """
    Main program.
    """
    extract_lexicon("res/twitter-2013train-A.txt", "svm_in/train.svm")
    extract_lexicon("res/twitter-2013dev-A.txt", "svm_in/dev.svm")
    extract_lexicon("res/twitter-2013test-A.txt", "svm_in/test.svm")

    # with open("lexicon.txt", "w") as f:
    #     for idx, w in enumerate(lexicon.keys()):
    #         f.write(f"{idx + 1}\t{w}\n")


if __name__ == "__main__":
    main()
