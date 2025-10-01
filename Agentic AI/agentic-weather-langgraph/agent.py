"""Weather agent implementation"""

import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langgraph.graph import StateGraph, START, END  # type: ignore

from llm_provider import LLMProvider
from tools import WeatherTool, SearchTool, SimpleRAG
from config import config


# Agent State
class AgentState(TypedDict):
    query: str
    plan: str
    tools: List[str]
    results: Dict[str, Any]
    answer: str
    feedback: Optional[str]


class QueryParser:
    """Query parser for weather requests"""
    
    @staticmethod
    def parse(query: str, default_days: int = None) -> Tuple[str, int, str]:
        """Extract location, days, and units from query"""
        default_days = default_days or config.default_forecast_days
        query = (query or "").strip().lower()
        
        # Check for coordinates first
        coords_match = re.match(r"^\s*([-+]?\d{1,3}\.\d+)\s*,\s*([-+]?\d{1,3}\.\d+)\s*", query)
        if coords_match:
            # Extract days and units from the rest of the query
            remaining = query[coords_match.end():].strip()
            days = default_days
            if "tomorrow" in remaining:
                days = 1
            else:
                days_match = re.search(r"for\s+(\d{1,2})\s+day", remaining)
                if days_match:
                    days = max(config.min_forecast_days, min(config.max_forecast_days, int(days_match.group(1))))
            
            units = "imperial" if any(word in remaining for word in ["fahrenheit", "imperial"]) else "metric"
            return f"{coords_match.group(1)},{coords_match.group(2)}", days, units
        
        # Extract days
        days = default_days
        if "tomorrow" in query:
            days = 1
        else:
            days_match = re.search(r"for\s+(\d{1,2})\s+day", query)
            if days_match:
                days = max(config.min_forecast_days, min(config.max_forecast_days, int(days_match.group(1))))
        
        # Extract units
        units = "imperial" if any(word in query for word in ["fahrenheit", "imperial"]) else "metric"
        
        # Extract location - simplified approach
        location = query
        # Remove common weather-related words and time references
        location = re.sub(r"\b(weather|for|tomorrow|days?|fahrenheit|imperial|celsius|metric|\d+|any|storm|alerts?|rain|cyclone|hurricane|news|emergency)\b", "", location)
        location = re.sub(r"\bin\s+", "", location)  # Remove "in" prefix
        location = re.sub(r"\s+", " ", location)  # Normalize whitespace
        location = location.strip(" ,?")
        
        return location or "current location", days, units


class WeatherAgent:
    """Weather agent"""
    
    def __init__(self, llm_provider: str = "auto"):
        self.llm = LLMProvider(provider=llm_provider)
        self.weather_tool = WeatherTool()
        self.search_tool = SearchTool()
        self.rag = SimpleRAG()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("plan", self._plan_node)
        graph.add_node("choose_tools", self._choose_tools_node)
        graph.add_node("run_tools", self._run_tools_node)
        graph.add_node("respond", self._respond_node)
        graph.add_node("learn", self._learn_node)
        
        # Add edges
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "choose_tools")
        graph.add_edge("choose_tools", "run_tools")
        graph.add_edge("run_tools", "respond")
        graph.add_edge("respond", "learn")
        graph.add_edge("learn", END)
        
        return graph.compile()
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Generate dynamic plan through reasoning"""
        state["plan"] = self.llm.plan(state["query"])
        return state
    
    def _choose_tools_node(self, state: AgentState) -> AgentState:
        """Choose tools using LLM reasoning"""
        state["tools"] = self.llm.select_tools(state["query"])
        return state
    
    def _run_tools_node(self, state: AgentState) -> AgentState:
        """Execute selected tools"""
        results = {}
        
        # Parse query
        location, days, units = QueryParser.parse(state["query"])
        results["parsed"] = {"location": location, "days": days, "units": units}
        
        # Run tools
        if "weather" in state["tools"]:
            weather_result = self.weather_tool.execute(location=location, days=days, units=units)
            results["weather"] = {
                "ok": weather_result.ok,
                "data": weather_result.data
            }
        
        if "rag" in state["tools"]:
            rag_result = self.rag.execute(query=state["query"])
            results["rag"] = rag_result.data if rag_result.ok else []
        
        if "search" in state["tools"]:
            search_result = self.search_tool.execute(query=state["query"])
            results["search"] = {
                "ok": search_result.ok,
                "data": search_result.data if search_result.ok else []
            }
        
        state["results"] = results
        return state
    
    def _respond_node(self, state: AgentState) -> AgentState:
        """Generate LLM-based dynamic response"""
        context = {
            "query": state["query"],
            "plan": state["plan"],
            "tools_used": ', '.join(state["tools"]),
            "parsed": state["results"].get("parsed", {}),
            "weather_data": state["results"].get("weather", {}),
            "rag_notes": state["results"].get("rag", []),
            "search_results": state["results"].get("search", {}),
            "feedback": state.get("feedback", "")
        }
        
        state["answer"] = self.llm.respond(context)
        return state
    
    def _learn_node(self, state: AgentState) -> AgentState:
        """Learn from interaction using LLM reasoning"""
        interaction = {
            "query": state["query"],
            "results": state["results"],
            "tools_used": ', '.join(state["tools"])
        }
        
        learning_insight = self.llm.learn(interaction)
        self.rag.add(learning_insight, "learn")
        
        # Store feedback
        if state.get("feedback"):
            import json
            with open(config.feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"query": state["query"], "feedback": state["feedback"]}) + "\n")
        
        return state
    
    def run(self, query: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent with a query"""
        initial_state: AgentState = {
            "query": query,
            "plan": "",
            "tools": [],
            "results": {},
            "answer": "",
            "feedback": feedback
        }
        return self.graph.invoke(initial_state)
