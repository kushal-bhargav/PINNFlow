"""
Codal rule store.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - optional fallback
    TfidfVectorizer = None
    cosine_similarity = None


class CodalRuleStore:
    """
    Manages semantic indexing and retrieval of engineering rules.

    Retrieval is intentionally hybrid:
    - keyword indexing for fast exact hits
    - TF-IDF similarity over rule descriptions/excerpts for better recall
    - context-based boosts for pressure, temperature, and velocity queries
    """

    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.semantic_index: Dict[str, List[int]] = {}
        self._documents: List[str] = []
        self._vectorizer = None
        self._tfidf_matrix = None

    def add_rules(self, new_rules: List[Dict[str, Any]]) -> None:
        for rule in new_rules:
            self.rules.append(rule)
        self._rebuild_indexes()

    def query(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_str = str(context.get("query", "") or "").lower().strip()
        candidate_indexes = self._candidate_indexes(query_str, context)
        verbose = context.get("verbose", False)

        if not candidate_indexes:
            generic_rules = [rule for rule in self.rules if rule["code"] == "GENERIC_LAYOUT"]
            if verbose:
                print(f"[RuleStore] Query '{query_str}' returned {len(generic_rules)} generic rules.")
            return generic_rules

        ranked = sorted(
            candidate_indexes,
            key=lambda idx: self._score_rule(idx, query_str, context),
            reverse=True,
        )
        results = [self.rules[idx] for idx in ranked if self._score_rule(idx, query_str, context) > 0.0]

        if results:
            if verbose:
                print(f"[RuleStore] Query '{query_str}' returned {len(results)} relevant rules.")
            return results

        generic_rules = [rule for rule in self.rules if rule["code"] == "GENERIC_LAYOUT"]
        if verbose:
            print(f"[RuleStore] Query '{query_str}' returned {len(generic_rules)} generic rules.")
        return generic_rules

    def query_by_rule_type(self, rule_type: str) -> List[Dict[str, Any]]:
        matches = [rule for rule in self.rules if rule["rule_type"] == rule_type]
        # query_by_rule_type is usually only called during startup initialization,
        # but we guard it just in case.
        return matches

    def best_rule(
        self,
        context: Dict[str, Any],
        rule_type: str | None = None,
        code: str | None = None,
    ) -> Dict[str, Any] | None:
        """
        Return the most relevant rule for a context.

        Agents should pass in the current state-derived context so the selected
        provision reflects the actual design point rather than a fixed heuristic.
        """
        candidates = self.query(context)
        if code:
            for rule in candidates:
                if rule.get("code") == code:
                    if rule_type is None or rule.get("rule_type") == rule_type:
                        return rule
            for rule in self.rules:
                if rule.get("code") == code:
                    if rule_type is None or rule.get("rule_type") == rule_type:
                        return rule

        if rule_type:
            for rule in candidates:
                if rule.get("rule_type") == rule_type:
                    return rule
            for rule in self.rules:
                if rule.get("rule_type") == rule_type:
                    return rule

        return candidates[0] if candidates else None

    def list_codes(self) -> List[str]:
        return sorted({rule["code"] for rule in self.rules})

    def _rebuild_indexes(self) -> None:
        self.semantic_index = {}
        self._documents = []

        for idx, rule in enumerate(self.rules):
            document = self._rule_document(rule)
            self._documents.append(document)
            for keyword in self._tokenize(document):
                self.semantic_index.setdefault(keyword, []).append(idx)

        if TfidfVectorizer is None or not self._documents:
            self._vectorizer = None
            self._tfidf_matrix = None
            return

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._tfidf_matrix = self._vectorizer.fit_transform(self._documents)

    def _rule_document(self, rule: Dict[str, Any]) -> str:
        conditions = rule.get("conditions", {}) or {}
        condition_values = " ".join(str(value) for value in conditions.values())
        parts = [
            str(rule.get("code", "")),
            str(rule.get("id", "")),
            str(rule.get("rule_type", "")),
            str(rule.get("description", "")),
            str(rule.get("formula", "")),
            str(rule.get("source_excerpt", "")),
            condition_values,
        ]
        return " ".join(part for part in parts if part)

    def _candidate_indexes(self, query_str: str, context: Dict[str, Any]) -> set[int]:
        candidates: set[int] = set()

        for keyword in self._tokenize(query_str):
            if keyword in self.semantic_index:
                candidates.update(self.semantic_index[keyword])

        if query_str and self._vectorizer is not None and self._tfidf_matrix is not None:
            query_vector = self._vectorizer.transform([query_str])
            similarities = cosine_similarity(query_vector, self._tfidf_matrix).ravel()
            candidates.update(idx for idx, score in enumerate(similarities) if score > 0.0)

        if "pressure" in context:
            candidates.update(
                idx for idx, rule in enumerate(self.rules) if rule["rule_type"] in {"stress_limit", "pressure_rating"}
            )

        if "temperature" in context or "delta_T" in context:
            candidates.update(
                idx for idx, rule in enumerate(self.rules) if rule["rule_type"] == "temperature_limit"
            )

        if "velocity" in context or "flow" in context:
            candidates.update(
                idx for idx, rule in enumerate(self.rules) if rule["rule_type"] == "erosional_velocity"
            )

        return candidates

    def _score_rule(self, idx: int, query_str: str, context: Dict[str, Any]) -> float:
        rule = self.rules[idx]
        score = 0.0

        query_tokens = set(self._tokenize(query_str))
        rule_tokens = set(self._tokenize(self._documents[idx])) if idx < len(self._documents) else set()
        score += float(len(query_tokens & rule_tokens))

        if query_str and self._vectorizer is not None and self._tfidf_matrix is not None:
            query_vector = self._vectorizer.transform([query_str])
            tfidf_score = cosine_similarity(query_vector, self._tfidf_matrix[idx]).ravel()[0]
            score += tfidf_score * 5.0

        rule_type = rule.get("rule_type")
        if "pressure" in context and rule_type in {"stress_limit", "pressure_rating"}:
            score += 2.0
        if ("temperature" in context or "delta_T" in context) and rule_type == "temperature_limit":
            score += 1.5
        if ("velocity" in context or "flow" in context) and rule_type == "erosional_velocity":
            score += 2.0

        return score

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9_.]+", text.lower())
