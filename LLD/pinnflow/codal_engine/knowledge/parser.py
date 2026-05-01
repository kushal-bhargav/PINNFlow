"""
Codal rule parser.

Converts extracted standards text into structured JSON rules using
deterministic, section-aware heuristics.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


class CodalParser:
    """
    Extracts multiple engineering rules from raw standards text.

    The original implementation only emitted at most one ASME rule and one API
    rule per document. This parser instead scans for repeated rule-bearing
    passages, associates them with nearby section identifiers, and emits a
    structured rule object for each distinct match.
    """

    CODE_PATTERNS = {
        "ASME B31.3": (
            re.compile(r"\bASME\s*B31\.3\b"),
            re.compile(r"\bB31\.3\b"),
        ),
        "API 14E": (
            re.compile(r"\bAPI\s*14E\b"),
            re.compile(r"\b14E\b"),
        ),
    }

    SOURCE_FILES = {
        "ASME B31.3": "ASME_B31_3.pdf",
        "API 14E": "API_4E.pdf",
    }

    RULE_SPECS = (
        {
            "codes": {"ASME B31.3"},
            "rule_type": "stress_limit",
            "keywords": (
                r"ALLOWABLE STRESS",
                r"STRESS LIMIT",
                r"DESIGN STRESS",
                r"MAXIMUM STRESS",
                r"SHALL NOT EXCEED",
            ),
            "limit_patterns": (
                r"(?:ALLOWABLE STRESS|STRESS LIMIT|MAXIMUM STRESS|SHALL NOT EXCEED)[^0-9]{0,40}(\d+(?:\.\d+)?)",
                r"\bS\s*=\s*(\d+(?:\.\d+)?)\b",
            ),
            "default_limit": 200.0,
            "default_formula": "t = (P * D) / (2 * (S * E + P * Y))",
            "description": "Stress limit guidance extracted from codal text.",
        },
        {
            "codes": {"ASME B31.3"},
            "rule_type": "thickness_requirement",
            "keywords": (
                r"MINIMUM WALL THICKNESS",
                r"REQUIRED THICKNESS",
                r"WALL THICKNESS",
                r"PIPE THICKNESS",
            ),
            "limit_patterns": (
                r"(?:MINIMUM WALL THICKNESS|REQUIRED THICKNESS|WALL THICKNESS)[^0-9]{0,40}(\d+(?:\.\d+)?)",
                r"\bT\s*=\s*(\d+(?:\.\d+)?)\b",
            ),
            "default_limit": 0.0,
            "default_formula": "t = (P * D) / (2 * (S * E + P * Y))",
            "description": "Wall-thickness requirement extracted from codal text.",
        },
        {
            "codes": {"ASME B31.3"},
            "rule_type": "pressure_rating",
            "keywords": (
                r"DESIGN PRESSURE",
                r"PRESSURE RATING",
                r"PRESSURE LIMIT",
                r"MAXIMUM ALLOWABLE PRESSURE",
            ),
            "limit_patterns": (
                r"(?:DESIGN PRESSURE|PRESSURE RATING|PRESSURE LIMIT|MAXIMUM ALLOWABLE PRESSURE)[^0-9]{0,40}(\d+(?:\.\d+)?)",
                r"\bP\s*=\s*(\d+(?:\.\d+)?)\b",
            ),
            "default_limit": 0.0,
            "default_formula": "P = (2 * S * E * t) / (D - 2 * Y * t)",
            "description": "Pressure-rating requirement extracted from codal text.",
        },
        {
            "codes": {"ASME B31.3"},
            "rule_type": "temperature_limit",
            "keywords": (
                r"DESIGN TEMPERATURE",
                r"TEMPERATURE LIMIT",
                r"MAXIMUM TEMPERATURE",
            ),
            "limit_patterns": (
                r"(?:DESIGN TEMPERATURE|TEMPERATURE LIMIT|MAXIMUM TEMPERATURE)[^0-9]{0,40}(\d+(?:\.\d+)?)",
            ),
            "default_limit": 0.0,
            "default_formula": "",
            "description": "Temperature limit extracted from codal text.",
        },
        {
            "codes": {"ASME B31.3"},
            "rule_type": "corrosion_allowance",
            "keywords": (
                r"CORROSION ALLOWANCE",
                r"EROSION ALLOWANCE",
            ),
            "limit_patterns": (
                r"(?:CORROSION ALLOWANCE|EROSION ALLOWANCE)[^0-9]{0,40}(\d+(?:\.\d+)?)",
            ),
            "default_limit": 0.0,
            "default_formula": "",
            "description": "Corrosion allowance extracted from codal text.",
        },
        {
            "codes": {"API 14E"},
            "rule_type": "erosional_velocity",
            "keywords": (
                r"EROSIONAL VELOCITY",
                r"VELOCITY LIMIT",
                r"VELOCITY SHALL NOT EXCEED",
                r"\bVE\s*=",
                r"\bC\s*=",
            ),
            "limit_patterns": (
                r"(?:EROSIONAL VELOCITY|VELOCITY LIMIT|VELOCITY SHALL NOT EXCEED)[^0-9]{0,40}(\d+(?:\.\d+)?)",
                r"\bC\s*=\s*(\d+(?:\.\d+)?)\b",
            ),
            "default_limit": 60.0,
            "default_formula": "Ve = C / sqrt(rho)",
            "description": "Erosional-velocity requirement extracted from codal text.",
        },
    )

    def parse_text_to_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract rules from standards text.

        The parser keeps enough local context around matched phrases to preserve
        section IDs and nearby formulas without requiring a full NLP stack.
        """
        prepared_text = self._prepare_text(text)
        if not prepared_text:
            return []

        rules: List[Dict[str, Any]] = []
        for code in self._detect_codes(prepared_text):
            rules.extend(self._extract_rules_for_code(code, prepared_text))
        return self._deduplicate_rules(rules)

    def validate_rule(self, rule: Dict[str, Any]) -> bool:
        """Ensures the rule follows the required schema."""
        required = ["code", "id", "rule_type", "description", "limit", "conditions"]
        return all(key in rule for key in required)

    def _prepare_text(self, text: str) -> str:
        upper = text.upper().replace("\r\n", "\n").replace("\r", "\n")
        upper = re.sub(r"[ \t]+", " ", upper)
        return re.sub(r"\n{3,}", "\n\n", upper).strip()

    def _detect_codes(self, text: str) -> List[str]:
        matches: List[str] = []
        for code, patterns in self.CODE_PATTERNS.items():
            if any(pattern.search(text) for pattern in patterns):
                matches.append(code)
        return matches

    def _extract_rules_for_code(self, code: str, text: str) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        section_spans = self._section_spans(text, code)
        for spec in self.RULE_SPECS:
            if code not in spec["codes"]:
                continue

            keyword_matches = list(self._find_keyword_matches(text, spec["keywords"]))
            if not keyword_matches:
                continue

            for ordinal, match in enumerate(keyword_matches, start=1):
                section_id, chunk = self._section_chunk(
                    text=text,
                    anchor_start=match.start(),
                    anchor_end=match.end(),
                    fallback_ordinal=ordinal,
                    section_spans=section_spans,
                )
                limit = self._extract_limit(chunk, spec["limit_patterns"], spec["default_limit"])
                formula = self._extract_formula(chunk, spec["default_formula"])
                excerpt = self._compact_excerpt(chunk)
                rules.append(
                    {
                        "code": code,
                        "id": self._build_rule_id(code, section_id, spec["rule_type"]),
                        "rule_type": spec["rule_type"],
                        "description": self._build_description(spec["description"], section_id, excerpt),
                        "formula": formula,
                        "limit": limit,
                        "conditions": {
                            "source": self.SOURCE_FILES.get(code, ""),
                            "section": section_id,
                            "hierarchy": [section_id],
                            "matched_text": self._matched_keyword(match.group(0)),
                        },
                        "source_excerpt": excerpt,
                    }
                )

        if rules:
            return rules

        return self._fallback_rules(code, text)

    def _find_keyword_matches(self, text: str, keywords: Iterable[str]) -> Iterable[re.Match[str]]:
        seen_spans = set()
        for keyword in keywords:
            for match in re.finditer(keyword, text):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                yield match

    def _context_window(self, text: str, start: int, end: int, radius: int = 240) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return text[left:right]

    def _section_spans(self, text: str, code: str) -> List[tuple[str, int, int]]:
        matches = list(self._section_pattern(code).finditer(text))
        spans: List[tuple[str, int, int]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            spans.append((match.group(1), start, end))
        return spans

    def _section_chunk(
        self,
        text: str,
        anchor_start: int,
        anchor_end: int,
        fallback_ordinal: int,
        section_spans: List[tuple[str, int, int]],
    ) -> tuple[str, str]:
        for section_id, start, end in section_spans:
            if start <= anchor_start < end:
                return section_id, text[start:end].strip()
        return f"GENERAL-{fallback_ordinal}", self._context_window(text, anchor_start, anchor_end)

    def _section_pattern(self, code: str) -> re.Pattern[str]:
        if code == "API 14E":
            return re.compile(r"\b(?:SECTION\s+)?((?:\d+\.)+\d+[A-Z]?|\d+\.\d+[A-Z]?)\b")
        return re.compile(r"\b(?:SECTION\s+)?((?:\d+\.)+\d+)\b")

    def _extract_limit(self, chunk: str, patterns: Iterable[str], default: float) -> float:
        for pattern in patterns:
            match = re.search(pattern, chunk)
            if not match:
                continue
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
        return default

    def _extract_formula(self, chunk: str, default: str) -> str:
        formula_match = re.search(r"\b([A-Z][A-Z0-9]*)\s*=\s*([A-Z0-9\(\)\*/\+\-\.\s]+)", chunk)
        if formula_match:
            left = formula_match.group(1).strip().title()
            right = re.sub(r"\s+", " ", formula_match.group(2)).strip(" .;,:")
            return f"{left} = {right}"
        return default

    def _compact_excerpt(self, text: str, limit: int = 220) -> str:
        excerpt = re.sub(r"\s+", " ", text).strip()
        if len(excerpt) <= limit:
            return excerpt
        return excerpt[: limit - 3].rstrip() + "..."

    def _matched_keyword(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _build_rule_id(self, code: str, section_id: str, rule_type: str) -> str:
        code_slug = re.sub(r"[^A-Z0-9]+", "-", code.upper()).strip("-")
        return f"{code_slug}-{section_id}-{rule_type}"

    def _build_description(self, base: str, section_id: str, excerpt: str) -> str:
        if section_id.startswith("GENERAL-"):
            return f"{base} Context: {excerpt}"
        return f"{base} Section {section_id}. Context: {excerpt}"

    def _fallback_rules(self, code: str, text: str) -> List[Dict[str, Any]]:
        if code == "ASME B31.3":
            return [
                {
                    "code": code,
                    "id": self._build_rule_id(code, "GENERAL-1", "stress_limit"),
                    "rule_type": "stress_limit",
                    "description": "Stress limit guidance extracted from parsed standards text.",
                    "formula": "t = (P * D) / (2 * (S * E + P * Y))",
                    "limit": 200.0,
                    "conditions": {
                        "source": self.SOURCE_FILES[code],
                        "section": "GENERAL-1",
                        "hierarchy": ["GENERAL-1"],
                        "matched_text": "ASME B31.3",
                    },
                    "source_excerpt": self._compact_excerpt(text),
                }
            ]

        if code == "API 14E":
            return [
                {
                    "code": code,
                    "id": self._build_rule_id(code, "GENERAL-1", "erosional_velocity"),
                    "rule_type": "erosional_velocity",
                    "description": "Erosional velocity guidance extracted from parsed standards text.",
                    "formula": "Ve = C / sqrt(rho)",
                    "limit": 60.0,
                    "conditions": {
                        "source": self.SOURCE_FILES[code],
                        "section": "GENERAL-1",
                        "hierarchy": ["GENERAL-1"],
                        "matched_text": "API 14E",
                    },
                    "source_excerpt": self._compact_excerpt(text),
                }
            ]

        return []

    def _deduplicate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for rule in rules:
            key = (
                rule["code"],
                rule["rule_type"],
                rule["conditions"].get("section"),
                round(float(rule.get("limit", 0.0)), 6),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rule)
        return deduped
