"""
Causal DAG Topology for Ecommerce KPIs
=======================================
Defines the directed acyclic graph structure where each node is a KPI
and each edge represents a causal relationship with a trained model.

DAG Layers:
  Layer 0 (Root Inputs):  ad_spend, CPC
  Layer 1 (Traffic):      impressions, clicks, CTR
  Layer 2 (Engagement):   sessions, new_users, bounce_rate, engaged_sessions
  Layer 3 (Conversion):   add_to_cart, conversions, AOV
  Layer 4 (Target):       revenue
  Layer 5 (Derived):      roas  (computed, not modeled)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass
class KPINode:
    """Single KPI in the causal graph."""

    id: str
    label: str
    unit: str
    layer: int
    parents: List[str] = field(default_factory=list)
    is_root: bool = False  # User can set directly (no parents)
    is_derived: bool = False  # Computed via formula, not a model
    is_target: bool = False  # Final prediction target
    min_val: float = 0.0
    max_val: float = float("inf")
    description: str = ""


# ── FULL DAG DEFINITION ──────────────────────────────────────────
DAG_NODES: Dict[str, KPINode] = {
    # Layer 0 — Root inputs (user-controlled, no upstream parents)
    "ad_spend": KPINode(
        id="ad_spend",
        label="Ad Spend",
        unit="₹",
        layer=0,
        parents=[],
        is_root=True,
        min_val=0,
        max_val=1_000_000,
        description="Total daily advertising budget across all channels",
    ),
    "CPC": KPINode(
        id="CPC",
        label="Cost Per Click",
        unit="₹",
        layer=0,
        parents=[],
        is_root=True,
        min_val=0.5,
        max_val=50,
        description="Average cost per ad click (platform-dependent)",
    ),
    # Layer 1 — Traffic (predicted from spend inputs)
    "impressions": KPINode(
        id="impressions",
        label="Impressions",
        unit="",
        layer=1,
        parents=["ad_spend", "CPC"],
        min_val=0,
        max_val=5_000_000,
        description="Total ad impressions served",
    ),
    "clicks": KPINode(
        id="clicks",
        label="Clicks",
        unit="",
        layer=1,
        parents=["impressions", "ad_spend", "CPC"],
        min_val=0,
        max_val=500_000,
        description="Total ad clicks",
    ),
    "CTR": KPINode(
        id="CTR",
        label="CTR",
        unit="%",
        layer=1,
        parents=["clicks", "impressions"],
        min_val=0.1,
        max_val=15,
        description="Click-through rate = clicks / impressions × 100",
    ),
    # Layer 2 — Engagement (predicted from traffic)
    "sessions": KPINode(
        id="sessions",
        label="Sessions",
        unit="",
        layer=2,
        parents=["clicks", "impressions"],
        min_val=0,
        max_val=200_000,
        description="Website sessions from all traffic sources",
    ),
    "new_users": KPINode(
        id="new_users",
        label="New Users",
        unit="",
        layer=2,
        parents=["sessions", "ad_spend"],
        min_val=0,
        max_val=100_000,
        description="First-time visitors",
    ),
    "bounce_rate": KPINode(
        id="bounce_rate",
        label="Bounce Rate",
        unit="%",
        layer=2,
        parents=["sessions", "CTR"],
        min_val=5,
        max_val=95,
        description="% of single-page sessions (lower is better)",
    ),
    # Layer 3 — Conversion (predicted from engagement)
    "conversions": KPINode(
        id="conversions",
        label="Conversions",
        unit="",
        layer=3,
        # parents=["add_to_cart", "bounce_rate", "engaged_sessions"], min_val=0, max_val=20_000,
        parents=["new_users", "sessions", "bounce_rate"],
        min_val=0,
        max_val=20_000,
        description="Completed purchases",
    ),
    "AOV": KPINode(
        id="AOV",
        label="AOV",
        unit="₹",
        layer=3,
        parents=["new_users", "sessions"],
        min_val=100,
        max_val=20_000,
        description="Average order value",
    ),
    # Layer 4 — Target
    "revenue": KPINode(
        id="revenue",
        label="Revenue",
        unit="₹",
        layer=4,
        parents=["conversions", "AOV"],
        is_target=True,
        min_val=0,
        max_val=50_000_000,
        description="Total revenue = conversions × AOV",
    ),
    # Layer 5 — Derived (formula, not ML)
    "roas": KPINode(
        id="roas",
        label="ROAS",
        unit="x",
        layer=5,
        parents=["revenue", "ad_spend"],
        is_derived=True,
        min_val=0,
        max_val=100,
        description="Return on Ad Spend = revenue / ad_spend",
    ),
}


def get_topological_order() -> List[str]:
    """Return KPI IDs in topological order (parents before children)."""
    visited = set()
    order = []

    def visit(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)
        node = DAG_NODES[node_id]
        for parent_id in node.parents:
            visit(parent_id)
        order.append(node_id)

    for node_id in DAG_NODES:
        visit(node_id)

    return order


def get_children(node_id: str) -> List[str]:
    """Get all direct children of a node."""
    return [nid for nid, node in DAG_NODES.items() if node_id in node.parents]


def get_descendants(node_id: str) -> List[str]:
    """Get all descendants (children, grandchildren, etc.) of a node."""
    descendants = []
    queue = get_children(node_id)
    visited = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        descendants.append(current)
        queue.extend(get_children(current))
    return descendants


def get_ancestors(node_id: str) -> List[str]:
    """Get all ancestors (parents, grandparents, etc.) of a node."""
    ancestors = []
    queue = list(DAG_NODES[node_id].parents)
    visited = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        ancestors.append(current)
        queue.extend(DAG_NODES[current].parents)
    return ancestors


def get_layers() -> Dict[int, List[str]]:
    """Group nodes by their layer number."""
    layers = defaultdict(list)
    for nid, node in DAG_NODES.items():
        layers[node.layer].append(nid)
    return dict(layers)


def validate_dag() -> Tuple[bool, List[str]]:
    """Validate the DAG has no cycles and all parents exist."""
    errors = []

    # Check all parent references exist
    for nid, node in DAG_NODES.items():
        for pid in node.parents:
            if pid not in DAG_NODES:
                errors.append(f"Node '{nid}' references non-existent parent '{pid}'")

    # Check for cycles using topological sort
    in_degree = {nid: 0 for nid in DAG_NODES}
    for nid, node in DAG_NODES.items():
        for pid in node.parents:
            if pid in in_degree:
                in_degree[nid] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited_count = 0

    while queue:
        current = queue.pop(0)
        visited_count += 1
        for child_id in get_children(current):
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    if visited_count != len(DAG_NODES):
        errors.append("CYCLE DETECTED — DAG contains circular dependencies")

    # Check root nodes have no parents
    for nid, node in DAG_NODES.items():
        if node.is_root and len(node.parents) > 0:
            errors.append(f"Root node '{nid}' should not have parents")

    is_valid = len(errors) == 0
    return is_valid, errors
