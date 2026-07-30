"""C1: Real IntentRouter.

Rule-based with optional embedding-aware logistic regression fallback.
The classifier can be trained from labeled examples and persisted.
"""

import json

import numpy as np

from lumen.lumen.controller import TwinForceController

_TRAINED_CACHE: dict[str, dict] = {}


class IntentRouter:
    """Intent classifier with embedding-aware training capability."""

    def __init__(self, model_path: str | None = None):
        self.model = None               # fastText model
        self.lr_coef: np.ndarray | None = None   # logistic regression weights
        self.lr_classes: list[str] | None = None
        self.lr_path: str | None = None
        if model_path:
            if model_path.endswith(".json"):
                self._load_lr(model_path)
            else:
                try:
                    import fasttext
                    self.model = fasttext.load_model(model_path)
                except Exception:
                    pass

    def _load_lr(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.lr_coef = np.array(data["coef"])
        self.lr_classes = data["classes"]
        self.lr_path = path

    def _save_lr(self, path: str) -> None:
        data = {"coef": self.lr_coef.tolist(), "classes": self.lr_classes}
        with open(path, "w") as f:
            json.dump(data, f)

    def train_lr(
        self,
        X: np.ndarray,
        y: list[str],
        lr: float = 0.01,
        epochs: int = 200,
        save_path: str | None = None,
    ) -> None:
        """Train a multi-class logistic regression classifier.

        Args:
            X: (n_samples, n_features) embedding matrix.
            y: (n_samples,) string labels.
            lr: Learning rate.
            epochs: Number of training epochs.
            save_path: Optional path to persist the trained weights.
        """
        classes = sorted(set(y))
        n_classes = len(classes)
        n_features = X.shape[1]
        y_idx = np.array([classes.index(label) for label in y])

        # Shuffle
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(y))
        X, y_idx = X[indices], y_idx[indices]

        # Initialize weights
        coef = np.zeros((n_classes, n_features))

        for _ in range(epochs):
            logits = X @ coef.T
            logits = np.clip(logits, -100, 100)
            exps = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            gradient = probs
            gradient[np.arange(len(y_idx)), y_idx] -= 1.0
            coef -= lr * gradient.T @ X / len(y_idx)

        self.lr_coef = coef
        self.lr_classes = classes

        if save_path:
            self._save_lr(save_path)
            self.lr_path = save_path

    def classify(self, query: str, tfc: TwinForceController | None = None) -> str:
        q = query.lower().strip()

        # 1. Keyword rules (higher precision, keep as override)
        if any(
            q.startswith(prefix)
            for prefix in ("what is", "what are", "remember", "who is", "where is")
        ):
            return "factual"
        if any(prefix in q for prefix in ("why ", "how ", "explain")):
            return "exploratory"
        if any(prefix in q for prefix in ("connected to", "related to", "linked", "relation")):
            return "relational"
        if any(prefix in q for prefix in ("last ", "yesterday", "ago", "when ")):
            return "temporal"

        # 2. fastText model fallback
        if self.model is not None:
            try:
                label, prob = self.model.predict(q.replace("\n", " "))
                intent = label[0].replace("__label__", "")
                if prob[0] >= 0.7:
                    return intent
            except Exception:
                pass

        # 3. TFC deterministic fallback
        if tfc is not None and tfc.state.a > 0.6:
            return "exploratory"
        return "factual"

    def classify_with_embedding(
        self, query_embedding: np.ndarray, tfc: TwinForceController | None = None
    ) -> str:
        """Classify using trained LR weights and query embedding.

        Falls back to lexical classify() if no trained weights are available.
        """
        if self.lr_coef is not None and self.lr_classes is not None:
            logits = query_embedding.reshape(1, -1) @ self.lr_coef.T
            idx = int(np.argmax(logits))
            return self.lr_classes[idx]
        return self.classify("", tfc)

    def get_trained_labels(self) -> list[str] | None:
        """Return trained class labels, or None if untrained."""
        return self.lr_classes