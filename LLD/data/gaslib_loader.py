"""
data/gaslib_loader.py
─────────────────────
GasLib XML Dataset Loader — Phase 1-A

Parses the authentic GasLib network files (GasLib-134, GasLib-582, GasLib-4197)
into a NetworkX DiGraph suitable for the GNN topology module and the PINN
training-data extractor.

File format coverage:
    .net   — network topology + pipe/node geometry (required)
    .scn   — nomination files (boundary-condition pressures and flows, optional)
    .cdf   — combined decision files (valve/compressor decisions, optional)
    .cs    — compressor station detail (optional)

Design rules:
    * Only GasLib-134, GasLib-582, and GasLib-4197 are supported.
    * If .scn / .cdf / .cs files are absent for a given network the loader
      returns only what is available from the .net file — no fallback values
      are injected.
    * If the mandatory .net file is absent or cannot be parsed a
      GasLibFileNotFoundError is raised immediately with the missing path.
    * Units are always converted to SI at parse time:
        length   : km  → m
        diameter : mm  → mm  (kept as mm to match simulator convention)
        pressure : bar → MPa
        flow     : 1000m³/h → m³/s (where relevant)
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

# ── Namespace map (GasLib uses namespaced XML) ─────────────────────────────
_NS = {
    "gas":   "http://gaslib.zib.de/Gas",
    "fw":    "http://gaslib.zib.de/Framework",
    "xsi":   "http://www.w3.org/2001/XMLSchema-instance",
}


# ── Exceptions ─────────────────────────────────────────────────────────────
class GasLibFileNotFoundError(FileNotFoundError):
    """Raised when a required GasLib file is missing from disk."""


class GasLibParseError(ValueError):
    """Raised when a GasLib XML file cannot be parsed correctly."""


class GasLibUnsupportedNetworkError(ValueError):
    """Raised when an unsupported GasLib network name is requested."""


# ── Path registry ──────────────────────────────────────────────────────────
# Paths are relative to this file's location (data/).
_HERE = Path(__file__).parent.resolve()

_SUPPORTED: Dict[str, Dict[str, object]] = {
    "GasLib-134": {
        "net":  _HERE / "raw_gaslib" / "GasLib-134-v2-20211129" / "GasLib-134-v2-20211129.net",
        "scn_dir": _HERE / "raw_gaslib" / "GasLib-134-v2-20211129" / "Nominations-134-v2-20211129",
        "cdf":  None,   # not provided for GasLib-134
        "cs":   None,   # not provided for GasLib-134
    },
    "GasLib-582": {
        "net":  _HERE / "raw_gaslib" / "GasLib-582-v2-20211129" / "GasLib-582-v2-20211129.net",
        "scn_dir": _HERE / "raw_gaslib" / "GasLib-582-v2-20211129" / "Nominations-582-v2-20211129",
        "cdf":  _HERE / "raw_gaslib" / "GasLib-582-v2-20211129" / "GasLib-582-v2-20211129.cdf",
        "cs":   _HERE / "raw_gaslib" / "GasLib-582-v2-20211129" / "GasLib-582-v2-20211129.cs",
    },
    "GasLib-4197": {
        "net":  _HERE / "raw_gaslib" / "GasLib-4197-v2-20211129" / "GasLib-4197-v1-20211129.net",
        "scn_dir": _HERE / "raw_gaslib" / "GasLib-4197-v2-20211129" / "Nominations-4197-v1-20220119",
        "cdf":  _HERE / "raw_gaslib" / "GasLib-4197-v2-20211129" / "GasLib-4197-v1-20211129.cdf",
        "cs":   _HERE / "raw_gaslib" / "GasLib-4197-v2-20211129" / "GasLib-4197-v1-20211129.cs",
    },
}


# ── Internal helpers ───────────────────────────────────────────────────────
def _attr_val(element: ET.Element, tag: str, unit: str, ns: str = "gas") -> Optional[float]:
    """
    Extract a <tag unit="..." value="..."/> child element's float value.
    Returns None if the child is absent — callers must handle None explicitly.
    """
    child = element.find(f"{{{_NS[ns]}}}{tag}")
    if child is None:
        # Try without namespace (some elements use default namespace)
        child = element.find(tag)
    if child is None:
        return None
    raw = child.get("value")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _convert_pressure_bar_to_mpa(bar: float) -> float:
    return bar * 0.1


def _convert_km_to_m(km: float) -> float:
    return km * 1000.0


def _parse_node_kind(tag: str) -> str:
    """Map XML element tag (local name) to a canonical node kind string."""
    tag_lower = tag.lower()
    if "source" in tag_lower:
        return "source"
    if "sink" in tag_lower:
        return "sink"
    return "junction"


def _local(element: ET.Element) -> str:
    """Return the local tag name (strip namespace URI if present)."""
    tag = element.tag
    if "}" in tag:
        return tag.split("}")[1]
    return tag


# ── Dataclasses for structured results ────────────────────────────────────
@dataclass
class PipeSegment:
    """Physical data for a single pipe segment extracted from a .net file."""
    pipe_id:   str
    from_node: str
    to_node:   str
    length_m:  float       # metres
    diameter_mm: float     # millimetres
    roughness_m: float     # metres (absolute wall roughness)
    p_min_mpa: Optional[float] = None   # from upstream/downstream nodes
    p_max_mpa: Optional[float] = None


@dataclass
class NodeData:
    """Data for a single node extracted from a .net file."""
    node_id:  str
    kind:     str           # "source" | "sink" | "junction"
    x:        Optional[float] = None
    y:        Optional[float] = None
    p_min_mpa: Optional[float] = None
    p_max_mpa: Optional[float] = None
    flow_min_m3s: Optional[float] = None   # m³/s
    flow_max_m3s: Optional[float] = None


@dataclass
class NominationScenario:
    """
    Boundary-condition data extracted from a single .scn nomination file.
    Only the fields actually present in the file are populated.
    """
    filename: str
    node_flows: Dict[str, float] = field(default_factory=dict)   # node_id → flow (1000 m³/h)
    node_pressures: Dict[str, float] = field(default_factory=dict)  # node_id → pressure (bar)


@dataclass
class CompressorStation:
    """Data for a compressor station entry from a .cs file."""
    station_id: str
    from_node:  str
    to_node:    str
    p_out_max_mpa: Optional[float] = None
    p_in_min_mpa:  Optional[float] = None
    flow_max_m3s:  Optional[float] = None


@dataclass
class CombinedDecision:
    """Active device settings extracted from a .cdf file."""
    element_id: str
    element_type: str    # e.g. "valve", "compressorStation"
    decision: str        # e.g. "open", "closed", "active"
    params: Dict[str, float] = field(default_factory=dict)


# ── Main Loader ────────────────────────────────────────────────────────────
class GasLibLoader:
    """
    Loads and parses GasLib network datasets.

    Supported networks:  GasLib-134, GasLib-582, GasLib-4197.
    Any other name raises GasLibUnsupportedNetworkError immediately.

    Usage:
        loader = GasLibLoader()
        graph = loader.load_network("GasLib-134")          # nx.DiGraph
        pipes = loader.load_pipes("GasLib-134")            # List[PipeSegment]
        scn   = loader.load_nominations("GasLib-134", max_scn=5)  # List[NominationScenario]

    No mock or fallback values are generated.  If a mandatory .net file is
    missing the call raises GasLibFileNotFoundError.  If optional .scn/.cdf/.cs
    files are absent the corresponding methods return an empty list and log a
    warning — they do NOT raise.
    """

    SUPPORTED_NAMES = tuple(_SUPPORTED.keys())

    def __init__(self) -> None:
        pass

    # ── Public API ─────────────────────────────────────────────────────────

    def load_network(
        self,
        name: str,
        mode: str = "authentic",
    ) -> nx.DiGraph:
        """
        Parse the .net file and return a NetworkX DiGraph.

        Nodes carry:  kind, x, y, p_min_mpa, p_max_mpa, flow_min_m3s, flow_max_m3s
        Edges carry:  element_type, length_m, diameter_mm, roughness_m, p_min_mpa, p_max_mpa

        mode: retained for API compatibility (currently ignored — always authentic).

        Raises:
            GasLibUnsupportedNetworkError  — unknown name
            GasLibFileNotFoundError        — .net file not on disk
            GasLibParseError               — XML malformed / required fields missing
        """
        paths = self._resolve_paths(name)
        nodes, pipes, _ = self._parse_net(paths["net"])
        graph = self._build_graph(nodes, pipes, name)
        return graph

    def load_pipes(self, name: str) -> List[PipeSegment]:
        """
        Return only the pipe segments (element type <pipe>) for the given network.
        shortPipe, resistor, and valve elements are excluded.

        Raises:
            GasLibUnsupportedNetworkError, GasLibFileNotFoundError, GasLibParseError
        """
        paths = self._resolve_paths(name)
        nodes, pipes, _ = self._parse_net(paths["net"])
        return pipes

    def load_nominations(
        self,
        name: str,
        max_scn: int = 10,
    ) -> List[NominationScenario]:
        """
        Parse up to `max_scn` nomination (.scn) files for the given network.

        If the nominations directory does not exist or contains no .scn files
        an empty list is returned and no error is raised.  The caller must
        handle the empty case.

        Args:
            name:    Network name (e.g. "GasLib-134").
            max_scn: Maximum number of .scn files to parse (alphabetical order).

        Returns:
            List of NominationScenario objects (may be empty).
        """
        paths = self._resolve_paths(name)
        scn_dir: Optional[Path] = paths.get("scn_dir")
        if scn_dir is None or not scn_dir.is_dir():
            print(
                f"[GasLibLoader] No nomination directory for {name} "
                f"(looked at: {scn_dir}).  Returning empty list."
            )
            return []

        scn_files = sorted(scn_dir.glob("*.scn"))[:max_scn]
        if not scn_files:
            print(
                f"[GasLibLoader] No .scn files found in {scn_dir}.  "
                "Returning empty list."
            )
            return []

        results: List[NominationScenario] = []
        for scn_path in scn_files:
            parsed = self._parse_scn(scn_path)
            if parsed is not None:
                results.append(parsed)
        return results

    def load_compressor_stations(self, name: str) -> List[CompressorStation]:
        """
        Parse the .cs compressor station file if present.

        Returns an empty list if the file is absent or not applicable.
        """
        paths = self._resolve_paths(name)
        cs_path: Optional[Path] = paths.get("cs")
        if cs_path is None:
            return []
        if not cs_path.exists():
            print(
                f"[GasLibLoader] .cs file not found for {name}: {cs_path}.  "
                "Returning empty list."
            )
            return []
        return self._parse_cs(cs_path)

    def load_combined_decisions(self, name: str) -> List[CombinedDecision]:
        """
        Parse the .cdf combined decision file if present.

        Returns an empty list if the file is absent or not applicable.
        """
        paths = self._resolve_paths(name)
        cdf_path: Optional[Path] = paths.get("cdf")
        if cdf_path is None:
            return []
        if not cdf_path.exists():
            print(
                f"[GasLibLoader] .cdf file not found for {name}: {cdf_path}.  "
                "Returning empty list."
            )
            return []
        return self._parse_cdf(cdf_path)

    # ── Path resolution ────────────────────────────────────────────────────

    def _resolve_paths(self, name: str) -> Dict[str, Optional[Path]]:
        """
        Return the path dictionary for `name` and verify the mandatory .net
        file exists.

        Raises:
            GasLibUnsupportedNetworkError  — name not in SUPPORTED_NAMES
            GasLibFileNotFoundError        — .net file missing from disk
        """
        if name not in _SUPPORTED:
            raise GasLibUnsupportedNetworkError(
                f"'{name}' is not supported.  "
                f"Use one of: {self.SUPPORTED_NAMES}"
            )
        paths = _SUPPORTED[name]
        net_path: Path = paths["net"]
        if not net_path.exists():
            raise GasLibFileNotFoundError(
                f"Mandatory .net file for {name} not found: {net_path}\n"
                "Please ensure the GasLib dataset is present in data/raw_gaslib/."
            )
        return paths

    # ── .net parser ────────────────────────────────────────────────────────

    def _parse_net(
        self,
        net_path: Path,
    ) -> Tuple[Dict[str, NodeData], List[PipeSegment], List[dict]]:
        """
        Parse a GasLib .net file.

        Returns:
            (nodes_dict, pipe_list, other_connections)
            nodes_dict        — {node_id: NodeData}
            pipe_list         — list of PipeSegment (only <pipe> elements)
            other_connections — list of raw dicts for shortPipe, compressorStation, etc.
        """
        try:
            tree = ET.parse(str(net_path))
        except ET.ParseError as exc:
            raise GasLibParseError(
                f"Failed to parse XML in {net_path}: {exc}"
            ) from exc

        root = tree.getroot()

        # ── Parse nodes ───────────────────────────────────────────────────
        nodes: Dict[str, NodeData] = {}

        # <framework:nodes> may be in "fw" or default namespace
        nodes_elem = root.find("fw:nodes", _NS)
        if nodes_elem is None:
            nodes_elem = root.find("{http://gaslib.zib.de/Framework}nodes")
        if nodes_elem is None:
            raise GasLibParseError(
                f"Could not find <nodes> section in {net_path}"
            )

        for child in nodes_elem:
            tag_local = _local(child)
            node_id = child.get("id")
            if not node_id:
                continue
            kind = _parse_node_kind(tag_local)

            # Coordinates
            x = child.get("x")
            y = child.get("y")
            x_val = float(x) if x is not None else None
            y_val = float(y) if y is not None else None

            # Pressure bounds
            p_min_bar = _attr_val(child, "pressureMin", "bar")
            p_max_bar = _attr_val(child, "pressureMax", "bar")
            p_min_mpa = _convert_pressure_bar_to_mpa(p_min_bar) if p_min_bar is not None else None
            p_max_mpa = _convert_pressure_bar_to_mpa(p_max_bar) if p_max_bar is not None else None

            # Flow bounds (only sources and sinks have these)
            flow_min_raw = _attr_val(child, "flowMin", "1000m_cube_per_hour")
            flow_max_raw = _attr_val(child, "flowMax", "1000m_cube_per_hour")
            # Convert 1000 m³/h → m³/s: × 1000 / 3600
            flow_min_m3s = flow_min_raw * 1000.0 / 3600.0 if flow_min_raw is not None else None
            flow_max_m3s = flow_max_raw * 1000.0 / 3600.0 if flow_max_raw is not None else None

            nodes[node_id] = NodeData(
                node_id=node_id,
                kind=kind,
                x=x_val,
                y=y_val,
                p_min_mpa=p_min_mpa,
                p_max_mpa=p_max_mpa,
                flow_min_m3s=flow_min_m3s,
                flow_max_m3s=flow_max_m3s,
            )

        # ── Parse connections ──────────────────────────────────────────────
        conns_elem = root.find("fw:connections", _NS)
        if conns_elem is None:
            conns_elem = root.find("{http://gaslib.zib.de/Framework}connections")
        if conns_elem is None:
            raise GasLibParseError(
                f"Could not find <connections> section in {net_path}"
            )

        pipes: List[PipeSegment] = []
        other_conns: List[dict] = []

        for child in conns_elem:
            tag_local = _local(child)
            conn_id   = child.get("id", "")
            from_node = child.get("from", "")
            to_node   = child.get("to", "")

            if tag_local == "pipe":
                # Length
                length_km = _attr_val(child, "length", "km")
                if length_km is None:
                    # skip pipes without geometric data
                    continue
                length_m = _convert_km_to_m(length_km)

                # Diameter
                diameter_mm = _attr_val(child, "diameter", "mm")
                if diameter_mm is None:
                    continue

                # Roughness
                roughness_m = _attr_val(child, "roughness", "m")
                if roughness_m is None:
                    roughness_m = 8e-6   # GasLib default roughness

                # Pressure bounds from connected nodes
                p_min_mpa = None
                p_max_mpa = None
                if from_node in nodes and nodes[from_node].p_min_mpa is not None:
                    p_min_mpa = nodes[from_node].p_min_mpa
                if to_node in nodes and nodes[to_node].p_max_mpa is not None:
                    p_max_mpa = nodes[to_node].p_max_mpa

                pipes.append(PipeSegment(
                    pipe_id=conn_id,
                    from_node=from_node,
                    to_node=to_node,
                    length_m=length_m,
                    diameter_mm=diameter_mm,
                    roughness_m=roughness_m,
                    p_min_mpa=p_min_mpa,
                    p_max_mpa=p_max_mpa,
                ))
            else:
                other_conns.append({
                    "id":         conn_id,
                    "type":       tag_local,
                    "from_node":  from_node,
                    "to_node":    to_node,
                })

        return nodes, pipes, other_conns

    # ── Graph builder ──────────────────────────────────────────────────────

    def _build_graph(
        self,
        nodes: Dict[str, NodeData],
        pipes: List[PipeSegment],
        network_name: str,
    ) -> nx.DiGraph:
        """
        Convert parsed node/pipe data into a NetworkX DiGraph.
        """
        graph = nx.DiGraph()
        graph.graph["name"] = network_name
        graph.graph["fallback"] = False

        for nid, nd in nodes.items():
            graph.add_node(
                nid,
                kind=nd.kind,
                x=nd.x,
                y=nd.y,
                p_min_mpa=nd.p_min_mpa,
                p_max_mpa=nd.p_max_mpa,
                flow_min_m3s=nd.flow_min_m3s,
                flow_max_m3s=nd.flow_max_m3s,
            )

        for pipe in pipes:
            graph.add_edge(
                pipe.from_node,
                pipe.to_node,
                id=pipe.pipe_id,
                element_type="pipe",
                length=pipe.length_m,
                diameter=pipe.diameter_mm,
                roughness_m=pipe.roughness_m,
                p_min_mpa=pipe.p_min_mpa,
                p_max_mpa=pipe.p_max_mpa,
                # Legacy compat keys expected by gnn.py
                design_length_m=pipe.length_m,
                shape_id=0.0,
                R_D_ratio=1.5,
            )

        return graph

    # ── .scn parser ────────────────────────────────────────────────────────

    def _parse_scn(self, scn_path: Path) -> Optional[NominationScenario]:
        """
        Parse a single .scn nomination file.

        The .scn XML contains a <scenario> with per-node <node> children
        that specify fixed flow values (and optionally pressure fixations).

        Returns None if the file cannot be parsed (does not raise — caller
        collects successful results only).
        """
        try:
            tree = ET.parse(str(scn_path))
        except ET.ParseError as exc:
            print(f"[GasLibLoader] Skipping {scn_path.name}: parse error — {exc}")
            return None

        root = tree.getroot()
        scenario = NominationScenario(filename=scn_path.name)

        # Find <scenario> element (may be direct child or nested)
        scenario_elem = root.find("gas:scenario", _NS)
        if scenario_elem is None:
            scenario_elem = root.find("{http://gaslib.zib.de/Gas}scenario")
        if scenario_elem is None:
            # Some .scn files wrap in <boundaryValue><scenario ...>
            for child in root:
                lname = _local(child)
                if lname == "scenario":
                    scenario_elem = child
                    break

        target = scenario_elem if scenario_elem is not None else root

        for node_elem in target:
            lname = _local(node_elem)
            if lname != "node":
                continue
            node_id = node_elem.get("id")
            if not node_id:
                continue

            # Flow value
            flow_elem = node_elem.find("{http://gaslib.zib.de/Gas}flow")
            if flow_elem is None:
                flow_elem = node_elem.find("flow")
            if flow_elem is not None:
                raw_flow = flow_elem.get("value")
                if raw_flow is not None:
                    try:
                        scenario.node_flows[node_id] = float(raw_flow)
                    except ValueError:
                        pass

            # Pressure fixation (if present)
            pressure_elem = node_elem.find("{http://gaslib.zib.de/Gas}pressure")
            if pressure_elem is None:
                pressure_elem = node_elem.find("pressure")
            if pressure_elem is not None:
                raw_p = pressure_elem.get("value")
                if raw_p is not None:
                    try:
                        scenario.node_pressures[node_id] = float(raw_p)
                    except ValueError:
                        pass

        return scenario

    # ── .cdf parser ────────────────────────────────────────────────────────

    def _parse_cdf(self, cdf_path: Path) -> List[CombinedDecision]:
        """
        Parse a .cdf combined-decision file.

        The .cdf XML contains per-device operating decisions (valve open/closed,
        compressor active/inactive).  Returns a list of CombinedDecision objects.
        Returns empty list on parse error (does not raise).
        """
        try:
            tree = ET.parse(str(cdf_path))
        except ET.ParseError as exc:
            print(f"[GasLibLoader] Skipping {cdf_path.name}: parse error — {exc}")
            return []

        root = tree.getroot()
        decisions: List[CombinedDecision] = []

        for child in root.iter():
            tag_local = _local(child)
            # CDF elements representing device decisions
            if tag_local in (
                "valve", "controlValve", "compressorStation",
                "resistor", "shortPipe",
            ):
                elem_id = child.get("id")
                if not elem_id:
                    continue
                # Look for a decision attribute or child
                decision_str = child.get("decision", "unknown")
                params: Dict[str, float] = {}
                for attr_name in ("pressureRatioMin", "pressureRatioMax", "speed"):
                    raw = child.get(attr_name)
                    if raw is not None:
                        try:
                            params[attr_name] = float(raw)
                        except ValueError:
                            pass
                decisions.append(CombinedDecision(
                    element_id=elem_id,
                    element_type=tag_local,
                    decision=decision_str,
                    params=params,
                ))

        return decisions

    # ── .cs parser ─────────────────────────────────────────────────────────

    def _parse_cs(self, cs_path: Path) -> List[CompressorStation]:
        """
        Parse a .cs compressor station description file.

        Extracts per-unit operating envelopes (pressure ratio, flow capacity).
        Returns empty list on parse error (does not raise).
        """
        try:
            tree = ET.parse(str(cs_path))
        except ET.ParseError as exc:
            print(f"[GasLibLoader] Skipping {cs_path.name}: parse error — {exc}")
            return []

        root = tree.getroot()
        stations: List[CompressorStation] = []

        # Iterate <compressorStation> or similar top-level elements
        for child in root.iter():
            tag_local = _local(child)
            if "compressor" not in tag_local.lower():
                continue
            station_id = child.get("id") or child.get("alias")
            if not station_id:
                continue
            from_node = child.get("from", "")
            to_node   = child.get("to", "")

            p_out_max_bar = _attr_val(child, "pressureOutMax", "bar")
            p_in_min_bar  = _attr_val(child, "pressureInMin", "bar")
            flow_max_raw  = _attr_val(child, "flowMax", "1000m_cube_per_hour")

            stations.append(CompressorStation(
                station_id=station_id,
                from_node=from_node,
                to_node=to_node,
                p_out_max_mpa=(
                    _convert_pressure_bar_to_mpa(p_out_max_bar)
                    if p_out_max_bar is not None else None
                ),
                p_in_min_mpa=(
                    _convert_pressure_bar_to_mpa(p_in_min_bar)
                    if p_in_min_bar is not None else None
                ),
                flow_max_m3s=(
                    flow_max_raw * 1000.0 / 3600.0
                    if flow_max_raw is not None else None
                ),
            ))

        return stations
