from __future__ import annotations

from dotenv import load_dotenv

from langgraph_mini_swe_poc.graph import build_graph

load_dotenv()

graph = build_graph("single")
multi_graph = build_graph("multi")
negotiation_graph = build_graph("negotiate")
dual_swe_graph = build_graph("dual-swe")
loop_graph = build_graph("loop")
