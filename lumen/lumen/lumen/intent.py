"""C1: Real IntentRouter.

Simple rule-based router since fasttext model may not exist.
"""

from typing import Optional

from lumen.lumen.controller import TwinForceController


class IntentRouter:
    """Deterministic rule-based intent classifier with TFC fallback."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            try:
                import fasttext
                self.model = fasttext.load_model(model_path)
            except Exception:
                pass

    def classify(self, query: str, tfc: Optional[TwinForceController] = None) -> str:
        q = query.lower().strip()
        # Explicit rule routing
        if any(q.startswith(prefix) for prefix in ("what is", "what are", "remember", "who is", "where is")):
            return "factual"
        if any(prefix in q for prefix in ("why ", "how ", "explain")):
            return "exploratory"
        if any(prefix in q for prefix in ("connected to", "related to", "linked", "relation")):
            return "relational"
        if any(prefix in q for prefix in ("last ", "yesterday", "ago", "when ")):
            return "temporal"

        # fastText model fallback
        if self.model is not None:
            try:
                label, prob = self.model.predict(q.replace('\n', ' '))
                intent = label[0].replace("__label__", "")
                if prob[0] >= 0.7:
                    return intent
            except Exception:
                pass

        # TFC deterministic fallback
        if tfc is not None:
            if tfc.state.a > 0.6:
                return "exploratory"
        return "factual"
