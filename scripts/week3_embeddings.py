from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate

from ai4h.models.embeddings import WordEmbeddings, load_txt_embeddings

emb_file = Path(
    "data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
)
vocab, vectors = load_txt_embeddings(emb_file, max_vocab=100_000)
assert vectors.shape == (100_000, 50)

norms = np.linalg.norm(vectors, axis=1, keepdims=True)
unit_vectors = vectors / np.clip(norms, a_min=1e-12, a_max=None)
we = WordEmbeddings(unit_vectors, vocab)


def print_analogies():
    """print out similarities and analogies"""

    vec = we.vec
    most_similar = we.most_similar

    assert np.allclose(np.linalg.norm(vec("king")), 1)
    assert not np.allclose(np.linalg.norm(vec("king") + vec("car")), 1)

    for word in ["frog", "car", "company"]:
        print(f"Most similar {word}:")
        print(
            tabulate(
                most_similar(vec(word), topk=10, exclude=(word,)),
                headers=["word", "score"],
            )
        )

    analogies = [
        # king - man + woman ≈ queen
        ("king", "man", "woman", "queen"),
        # paris - france + italy ≈ rome
        ("paris", "france", "italy", "rome"),
        # berlin - germany + spain ≈ madrid
        ("berlin", "germany", "spain", "madrid"),
        # walking - walk + swim ≈ swimming
        ("walking", "walk", "swim", "swimming"),
    ]

    for c1, c2, c3, target in analogies:
        q = vec(c1) - vec(c2) + vec(c3)
        print(f"\nAnalogy: {c1} - {c2} + {c3} ≈ {target}")
        print(
            tabulate(
                most_similar(q, topk=10, exclude=(c1, c2, c3)),
                headers=["word", "score"],
            )
        )


def plot_analogy():
    # pairs
    pairs = [
        ("brother", "sister"),
        ("nephew", "niece"),
        ("uncle", "aunt"),
        ("man", "woman"),
        ("sir", "madam"),
        ("heir", "heiress"),
        ("king", "queen"),
        ("duke", "duchess"),
        ("earl", "countess"),
        ("emperor", "empress"),
    ]
    words = sorted([w for ab in pairs for w in ab])
    W = np.stack([we.vec(w) for w in words], axis=0)
    X2 = PCA(n_components=2, random_state=2025).fit_transform(W)
    coord = {w: xy for w, xy in zip(words, X2)}

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 10))
    # points + labels
    for w in words:
        x, y = coord[w]
        ax.scatter([x], [y], s=12)
        ax.text(x, y, f" {w}", fontsize=11, color="0.2", va="center")
    # dashed arrows e.g. masculine -> feminine
    for a, b in pairs:
        xa, ya = coord[a]
        xb, yb = coord[b]
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(xa, ya),
            arrowprops=dict(arrowstyle="-", linestyle="--", lw=1.5),
        )
    ax.set_title("Gender analogies in embedding space (masculine -> feminine)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("figs/gender_analogy.png")


def plot_embeddings():
    tsneModel = TSNE(n_components=2, random_state=0)
    n_v = 1000
    W_v = unit_vectors[:n_v]
    model2d = tsneModel.fit_transform(W_v)
    fig, ax = plt.subplots(figsize=(20, 20))
    for i, (x1, x2) in enumerate(model2d):
        w = vocab[i]
        ax.plot(x1, x2, "r.")
        ax.text(x1, x2, w)
    fig.tight_layout()
    fig.savefig(f"figs/embeddings_{n_v}.png")


def main():
    print_analogies()
    plot_embeddings()
    plot_analogy()


if __name__ == "__main__":
    main()
