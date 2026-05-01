from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnflow.deliverables.generator import DeliverableGenerator


def test_generate_iso_routes_by_shape(tmp_path):
    generator = DeliverableGenerator(output_dir=str(tmp_path))
    straight = generator.generate_iso("D-straight", [273.0, 9.27, 120.0, 5.0, 15.0, 20.0, 2.5, 0.55, 0.0, 0.9])
    elbow = generator.generate_iso("D-elbow", [273.0, 9.27, 120.0, 5.0, 15.0, 20.0, 2.5, 0.55, 1.0, 0.9])
    tee = generator.generate_iso("D-tee", [273.0, 9.27, 120.0, 5.0, 15.0, 20.0, 2.5, 0.55, 2.0, 0.9])

    with open(straight, "r", encoding="utf-8") as fh:
        straight_data = json.load(fh)
    with open(elbow, "r", encoding="utf-8") as fh:
        elbow_data = json.load(fh)
    with open(tee, "r", encoding="utf-8") as fh:
        tee_data = json.load(fh)

    assert straight_data["shape_id"] == 0
    assert elbow_data["shape_id"] == 1
    assert tee_data["shape_id"] == 2
    assert len(straight_data["segments"]) == 1
    assert len(elbow_data["segments"]) == 3
    assert len(tee_data["segments"]) == 3
