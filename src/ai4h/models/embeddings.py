from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_txt_embeddings(path: Path | str, max_vocab: int | None = None):
    vocab: list[str] = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split()
            # The embeddings are 50 dimensional
            word, vec = parts[:-50], parts[-50:]
            vocab.append("".join(word))
            vectors.append([float(x) for x in vec])
            if max_vocab and i + 1 >= max_vocab:
                break

    vectors = np.array(vectors, dtype=np.float32)
    return vocab, vectors


class WordEmbeddings:
    def __init__(self, unit_vectors: NDArray[np.float32], vocab: list[str]):
        self.unit_vectors = unit_vectors
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}

    def vec(self, word: str) -> NDArray[np.float32]:
        return self.unit_vectors[self.word2idx[word]]

    def most_similar(
        self,
        query_vec: NDArray[np.float32],
        topk: int = 10,
        exclude: tuple[str, ...] | None = None,
    ) -> list[tuple[str, float]]:
        q = query_vec / np.linalg.norm(query_vec)
        scores = self.unit_vectors @ q
        if exclude:
            for w in exclude:
                if w in self.word2idx:
                    scores[self.word2idx[w]] = -np.inf
        idx = np.argpartition(-scores, topk)[:topk]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.vocab[i], float(scores[i])) for i in idx]
