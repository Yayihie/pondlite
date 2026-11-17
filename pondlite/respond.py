"""Ground support answers in training records with stdlib TF-IDF retrieval."""

import math
import re
import unicodedata
from collections import Counter


ABSTAIN_MESSAGE = "Insufficient dataset information to answer this question safely."

_GREETING_PREFIX = re.compile(
    r"^(?:"
    r"hi\s+[^,!?.]+(?:\s*,\s*thanks\s+for\s+writing\s+in)?\s*[!?.]"
    r"|happy\s+to\s+help\s*[!?.]"
    r"|good\s+question\s*[!?.]"
    r"|thanks\s+for\s+reaching\s+out\s*[!?.]"
    r")\s*",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")


def _strip_greeting_prefixes(answer):
    """Remove leading support boilerplate without changing the policy text."""
    text = answer.strip()
    while text:
        stripped = _GREETING_PREFIX.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text


def normalize_answer(answer):
    """Normalize a policy answer for greeting-insensitive evaluation."""
    text = unicodedata.normalize("NFKC", _strip_greeting_prefixes(answer)).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _features(text):
    """Build word unigram and bigram features."""
    tokens = _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    return tokens + [tokens[i] + " " + tokens[i + 1] for i in range(len(tokens) - 1)]


def _normalized_tfidf(counts, idf):
    weights = {
        feature: count * idf[feature]
        for feature, count in counts.items()
        if feature in idf
    }
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if not norm:
        return {}
    return {feature: weight / norm for feature, weight in weights.items()}


class TfidfRetriever:
    """Retrieve and return the closest training answer, or explicitly abstain."""

    def __init__(self, records=None):
        self._records = []
        self._vectors = []
        self._idf = {}
        if records is not None:
            self.fit(records)

    def fit(self, records):
        """Fit word unigram+bigram TF-IDF vectors on instruction/input records."""
        records = list(records)
        texts = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("training records must be dictionaries")
            if not isinstance(record.get("instruction"), str) or not isinstance(record.get("output"), str):
                raise ValueError("training records require string instruction and output fields")
            text = record["instruction"]
            if record.get("input") is not None:
                if not isinstance(record["input"], str):
                    raise ValueError("training record input must be a string or null")
                text += "\n" + record["input"]
            texts.append(text)

        document_counts = [Counter(_features(text)) for text in texts]
        document_frequency = Counter()
        for counts in document_counts:
            for feature in counts:
                document_frequency[feature] += 1

        n_documents = len(records)
        idf = {
            feature: math.log((1 + n_documents) / (1 + frequency)) + 1.0
            for feature, frequency in document_frequency.items()
        }
        vectors = [_normalized_tfidf(counts, idf) for counts in document_counts]

        self._records = records
        self._idf = idf
        self._vectors = vectors
        return self

    def answer(self, question, input=None, threshold=0.45):
        """Return an answer, cosine confidence, matched source, and abstention flag."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if input is not None and not isinstance(input, str):
            raise TypeError("input must be a string or null")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        query = question if input is None else question + "\n" + input
        query_vector = _normalized_tfidf(Counter(_features(query)), self._idf)
        scores = [
            sum(weight * document.get(feature, 0.0) for feature, weight in query_vector.items())
            for document in self._vectors
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__) if scores else None
        confidence = scores[best_index] if best_index is not None else 0.0

        if best_index is None or confidence <= 0.0 or confidence < threshold:
            return {
                "answer": ABSTAIN_MESSAGE,
                "confidence": confidence,
                "matched_source": None,
                "abstained": True,
            }

        source = self._records[best_index]
        answer = _strip_greeting_prefixes(source["output"])
        if not answer:
            return {
                "answer": ABSTAIN_MESSAGE,
                "confidence": confidence,
                "matched_source": None,
                "abstained": True,
            }
        return {
            "answer": answer,
            "confidence": confidence,
            "matched_source": source,
            "abstained": False,
        }
