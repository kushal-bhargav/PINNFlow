from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinnflow.codal_engine.agents.asme_b31_agent import ASMEB31Agent
from pinnflow.codal_engine.agents.api14e_agent import API14EAgent
from pinnflow.codal_engine.knowledge.parser import CodalParser
from pinnflow.codal_engine.knowledge.rule_store import CodalRuleStore
from pinnflow.closed_loop.optimizer import ClosedLoopOptim


class _FakeEnv:
    BOUNDS = np.array(
        [
            [114, 620],
            [4, 22],
            [5, 150],
            [1, 20],
            [0, 150],
            [-40, 80],
            [0.5, 9],
            [0.3, 0.8],
            [0, 2],
            [0.3, 1.5],
        ],
        dtype=float,
    )


def _build_rule_store() -> CodalRuleStore:
    parser = CodalParser()
    store = CodalRuleStore()
    store.add_rules(parser.parse_text_to_rules("ASME B31.3 allowable stress limit 180 MPa"))
    store.add_rules(parser.parse_text_to_rules("API 14E erosional velocity limit 50 m/s"))
    return store


def test_codal_agents_emit_rule_values_and_recommendations():
    store = _build_rule_store()
    asme = ASMEB31Agent(store)
    api = API14EAgent(store)

    design = np.array([273.0, 8.0, 120.0, 12.0, 15.0, 20.0, 7.5, 0.55, 1.0, 1.2], dtype=float)
    context = {"sigma": 240.0, "velocity": 7.5, "pressure": 12.0, "delta_T": 20.0}

    asme_result = asme.evaluate(design, context)
    api_result = api.evaluate(design, context)

    assert asme_result["rule_value"]["limit"] > 0
    assert api_result["rule_value"]["limit"] > 0
    assert any(rec["parameter"] == "thickness" for rec in asme_result["recommendations"])
    assert any(rec["parameter"] == "diameter" for rec in api_result["recommendations"])


def test_codal_guidance_moves_design_toward_recommendations():
    store = _build_rule_store()
    env = _FakeEnv()
    dummy_agent = object()
    dummy_pinn = object()
    optimizer = ClosedLoopOptim(env=env, agent=dummy_agent, pinn=dummy_pinn)

    state = np.array([273.0, 8.0, 120.0, 12.0, 15.0, 20.0, 7.5, 0.55, 1.0, 1.2], dtype=float)
    info = {
        "codal_recommendations": [
            {"parameter": "thickness", "index": 1, "target": 10.0, "direction": "increase"},
            {"parameter": "diameter", "index": 0, "target": 300.0, "direction": "increase"},
        ]
    }

    guided = optimizer._apply_codal_guidance(state, info)

    assert guided[1] > state[1]
    assert guided[0] > state[0]
