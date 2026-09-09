"""Graph nodes. Each is ``(state, config) -> partial state``."""

from praxis.graph.nodes.analysts import bear, bull
from praxis.graph.nodes.editor import editor
from praxis.graph.nodes.planner import planner
from praxis.graph.nodes.red_team import red_team, route_after_red_team
from praxis.graph.nodes.researcher import researcher
from praxis.graph.nodes.verifier import verifier

__all__ = [
    "planner",
    "researcher",
    "bull",
    "bear",
    "red_team",
    "route_after_red_team",
    "editor",
    "verifier",
]
