"""
pinnflow/codal_engine/agents/base.py
───────────────────────────────────
MODULE 25 — Critique Agent Interface

Defines the base interface for specialized codal agents.
"""
from __future__ import annotations
import abc
from typing import Dict, Any, List
from pinnflow.codal_engine.knowledge.rule_store import CodalRuleStore

class CritiqueAgent(abc.ABC):
    """
    [V8] Industrial Critique Agent.
    Evaluates a design against specific engineering codes.
    """
    def __init__(self, rule_store: CodalRuleStore):
        self.rule_store = rule_store

    @abc.abstractmethod
    def evaluate(self, design: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs compliance check and returns penalty/status.
        """
        pass
