"""Graph nodes. Each is ``(state, config) -> partial state``."""

from praxis.graph.nodes.analysts import bear, bull
from praxis.graph.nodes.editor import editor
from praxis.graph.nodes.planner import planner
from praxis.graph.nodes.red_team import red_team, route_after_red_team
from praxis.graph.nodes.researcher import ResearchTask, dispatch_research, research_one
from praxis.graph.nodes.verifier import verifier

__all__ = [
    "planner",
    "dispatch_research",
    "research_one",
    "ResearchTask",
    "bull",
    "bear",
    "red_team",
    "route_after_red_team",
    "editor",
    "verifier",
]
