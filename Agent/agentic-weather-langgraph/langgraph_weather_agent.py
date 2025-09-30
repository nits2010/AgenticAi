from __future__ import annotations

import os
import sys
import re
import json
import argparse
import pathlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
import requests

# LLM integration for real agentic planning
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    # Fallback: simple .env loader if python-dotenv not available
    def load_dotenv():
        env_path = pathlib.Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    load_dotenv()

from langgraph.graph import StateGraph, START, END

# ----------------------- Configuration -----------------------
BASE_DIR = pathlib.Path.cwd()
MEM_DIR = BASE_DIR / "memory"
MEM_DIR.mkdir(exist_ok=True)
RAG_PATH = MEM_DIR / "notes.jsonl"
FEEDBACK_PATH = MEM_DIR / "feedback.jsonl"

# ----------------------- Simple RAG -----------------------
class SimpleRAG:
    def __init__(self):
        self._ensure_seeded()
    
    def _ensure_seeded(self):
        """Initialize with basic knowledge if file doesn't exist"""
        if not RAG_PATH.exists():
            self.add("Use local timezone and mention uncertainty after day 3.", "seed")
            self.add("Open-Meteo provides daily tmax/tmin, precip sum, wind gusts, sunrise/sunset.", "seed")
    
    def add(self, text: str, source: str = "user"):
        """Add knowledge to RAG"""
        with open(RAG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "source": source}) + "\n")
    
    def search(self, query: str, limit: int = 3) -> List[str]:
        """Search for relevant knowledge"""
        if not RAG_PATH.exists():
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for line in RAG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                data = json.loads(line)
                text = data.get("text", "")
                # Simple word overlap scoring
                score = sum(1 for word in text.lower().split() if word in query_words)
                if score > 0:
                    results.append((text, score))
            except json.JSONDecodeError:
                continue
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in results[:limit]]

# Global RAG instance
rag = SimpleRAG()

# ----------------------- Query Parser -----------------------
class QueryParser:
    """Simplified query parser for weather requests"""
    
    @staticmethod
    def parse(query: str, default_days: int = 3) -> Tuple[str, int, str]:
        """Extract location, days, and units from query"""
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
                    days = max(1, min(7, int(days_match.group(1))))
            
            units = "imperial" if any(word in remaining for word in ["fahrenheit", "imperial"]) else "metric"
            return f"{coords_match.group(1)},{coords_match.group(2)}", days, units
        
        # Extract days
        days = default_days
        if "tomorrow" in query:
            days = 1
        else:
            days_match = re.search(r"for\s+(\d{1,2})\s+day", query)
            if days_match:
                days = max(1, min(7, int(days_match.group(1))))
        
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

# ----------------------- Tools -----------------------
class ToolResult(TypedDict):
    name: str
    ok: bool
    data: Any
    meta: Dict[str, Any]


class LLMProvider:
    """Abstract LLM provider for multiple services"""
    
    def __init__(self, provider: str = "auto"):
        self.provider = self._detect_provider(provider)
        self.client = self._initialize_client()
    
    def _detect_provider(self, provider: str) -> str:
        """Auto-detect best available provider"""
        if provider == "auto":
            # Priority order: Groq (faster), then OpenAI
            if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
                return "groq"
            elif OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                return "openai"
            else:
                raise ValueError("No LLM provider available. Set GROQ_API_KEY or OPENAI_API_KEY in .env")
        return provider
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq library not installed. Install with: pip install groq")
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY not set in .env file")
            return groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not set in .env file")
            return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, messages: List[Dict[str, str]], model: str = None, temperature: float = 0.1, max_tokens: int = 800) -> str:
        """Generate response using the configured provider"""
        
        if self.provider == "groq":
            # Use Llama 3 by default for Groq
            model = model or "llama-3.1-8b-instant"
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        
        elif self.provider == "openai":
            # Use GPT-3.5 by default for OpenAI
            model = model or "gpt-3.5-turbo"
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_provider_info(self) -> str:
        """Get information about the current provider"""
        if self.provider == "groq":
            return "Groq (Llama 3.1-8b)"
        elif self.provider == "openai":
            return "OpenAI (GPT-3.5-turbo)"
        else:
            return f"Unknown provider: {self.provider}"


class WeatherTool:
    """Simplified weather tool using Open-Meteo API"""
    
    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    def _geocode(self, location: str) -> Optional[Dict[str, float]]:
        """Get coordinates for location"""
        # Handle direct coordinates
        if "," in location:
            try:
                lat, lon = map(float, location.split(",", 1))
                return {"name": location, "latitude": lat, "longitude": lon}
            except ValueError:
                pass
        
        # API geocoding
        try:
            response = requests.get(self.GEO_URL, params={"name": location, "count": 1}, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                result = results[0]
                return {
                    "name": result.get("name", location),
                    "latitude": result["latitude"],
                    "longitude": result["longitude"]
                }
        except Exception:
            pass
        return None

    def get_weather(self, location: str, days: int = 3, units: str = "metric") -> ToolResult:
        """Get weather forecast"""
        days = max(1, min(7, days))
        meta = {"location": location, "days": days, "units": units}

        # Get coordinates
        coords = self._geocode(location)
        if not coords:
            return {"name": "weather", "ok": False, "data": f"Could not find location: {location}", "meta": meta}
        
        # Get forecast
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "sunrise", "sunset"],
            "forecast_days": days,
            "temperature_unit": "fahrenheit" if units == "imperial" else "celsius",
            "wind_speed_unit": "mph" if units == "imperial" else "kmh",
        }
        
        try:
            response = requests.get(self.FORECAST_URL, params=params, timeout=12)
            response.raise_for_status()
            data = response.json()
            data["resolved"] = coords
            return {"name": "weather", "ok": True, "data": data, "meta": meta}
        except Exception as e:
            return {"name": "weather", "ok": False, "data": str(e), "meta": meta}

class SearchTool:
    """Simplified search tool using SerpAPI"""
    
    def is_available(self) -> bool:
        return bool(os.getenv("SERPAPI_API_KEY"))
    
    def search(self, query: str, limit: int = 3) -> ToolResult:
        """Search using SerpAPI"""
        if not self.is_available():
            return {"name": "search", "ok": False, "data": "SERPAPI_API_KEY not configured", "meta": {"query": query}}
        
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": os.getenv("SERPAPI_API_KEY")
                },
                timeout=10
            )
            response.raise_for_status()
            results = response.json().get("organic_results", [])[:limit]
            return {"name": "search", "ok": True, "data": results, "meta": {"query": query}}
        except Exception as e:
            return {"name": "search", "ok": False, "data": str(e), "meta": {"query": query}}

# ----------------------- Agent -----------------------
class AgentState(TypedDict):
    query: str
    plan: str
    tools: List[str]
    results: Dict[str, Any]
    answer: str
    feedback: Optional[str]

class WeatherAgent:
    """Simplified agentic weather agent"""
    
    def __init__(self, llm_provider: str = "auto"):
        self.weather_tool = WeatherTool()
        self.search_tool = SearchTool()
        self.llm = LLMProvider(provider=llm_provider)
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
        query = state["query"]
        
        plan = self._reason_about_plan(query)
        
        state["plan"] = plan
        return state
    
    def _reason_about_plan(self, query: str) -> str:
        
        # Use the abstracted LLM provider
        return self._call_llm_for_planning(query)
    
    def _call_llm_for_planning(self, query: str) -> str:
        """Pure LLM-based planning"""
        
        reasoning_prompt = f"""
You are an intelligent weather agent. Analyze this user query and generate a dynamic execution plan.

Query: "{query}"

Available tools:
- Weather API (Open-Meteo) for forecasts
- Search API (SerpAPI) for real-time alerts/news
- Knowledge base (RAG) for stored insights
- Geocoding for location resolution

Think step by step and generate a plan that shows your reasoning process:

1. **Query Analysis**: What is the user really asking for?
2. **Tool Selection**: What tools and data sources are needed?
3. **Strategy**: What's the most efficient approach?
4. **Plan Generation**: Create a step-by-step execution plan
5. **Reasoning Summary**: Explain your decisions

Format your response as:
**🧠 LLM-Generated Reasoning Process:**

**🔍 Query Analysis:**
[Your analysis of what the user wants]

**🎯 Dynamic Execution Plan:**
1. [Step 1]
2. [Step 2]
...

**💡 Reasoning Summary:**
[Your strategic decisions and why]

Be specific about why you chose certain tools and approaches.
"""
        
        messages = [
            {"role": "system", "content": "You are an intelligent weather agent planner. Be concise but thorough in your reasoning."},
            {"role": "user", "content": reasoning_prompt}
        ]
        
        llm_plan = self.llm.generate(messages, temperature=0.1, max_tokens=800)
        return f"{llm_plan}\n\n**🤖 Note:** This reasoning was generated by {self.llm.get_provider_info()}."
    
    
    
    def _choose_tools_node(self, state: AgentState) -> AgentState:
        """Choose tools using LLM reasoning"""
        
        # Use LLM to intelligently select tools
        tools = self._llm_choose_tools(state["query"])
        state["tools"] = tools
        return state
    
    def _llm_choose_tools(self, query: str) -> List[str]:
        """Use LLM to intelligently select tools based on query analysis"""
        
        tool_selection_prompt = f"""
You are an intelligent weather agent. Analyze this query and select the appropriate tools.

**User Query:** {query}

**Available Tools:**
- weather: Get weather forecasts and current conditions
- rag: Search knowledge base for stored insights and patterns
- search: Search for real-time weather alerts, news, and emergency information

**Tool Selection Guidelines:**
- Always include "weather" for weather-related queries
- Always include "rag" to check stored knowledge
- Include "search" for queries about alerts, storms, emergencies, news, or real-time information
- Consider the user's intent and what information they need

Return ONLY a comma-separated list of tool names (e.g., "weather,rag,search").
"""
        
        messages = [
            {"role": "system", "content": "You are a tool selection expert. Return only the tool names as requested."},
            {"role": "user", "content": tool_selection_prompt}
        ]
        
        tool_response = self.llm.generate(messages, temperature=0.1, max_tokens=100)
        
        # Parse the response and ensure we have valid tools
        selected_tools = [tool.strip() for tool in tool_response.split(",")]
        
        # Ensure we always have basic tools
        if "weather" not in selected_tools:
            selected_tools.append("weather")
        if "rag" not in selected_tools:
            selected_tools.append("rag")
        
        return selected_tools
    
    def _run_tools_node(self, state: AgentState) -> AgentState:
        """Execute selected tools"""
        results = {}
        
        # Parse query
        location, days, units = QueryParser.parse(state["query"])
        results["parsed"] = {"location": location, "days": days, "units": units}
        
        # Run tools
        if "weather" in state["tools"]:
            results["weather"] = self.weather_tool.get_weather(location, days, units)
        
        if "rag" in state["tools"]:
            results["rag"] = rag.search(state["query"])
        
        if "search" in state["tools"]:
            results["search"] = self.search_tool.search(state["query"])
        
        state["results"] = results
        return state
    
    def _respond_node(self, state: AgentState) -> AgentState:
        """Generate LLM-based dynamic response"""
        
        # Generate response using LLM reasoning
        response = self._generate_llm_response(state)
        state["answer"] = response
        return state
    
    def _generate_llm_response(self, state: AgentState) -> str:
        """Generate response using real LLM reasoning"""
        
        results = state["results"]
        query = state["query"]
        plan = state["plan"]
        
        # Prepare context for LLM
        context = {
            "query": query,
            "plan": plan,
            "tools_used": state["tools"],
            "parsed": results.get("parsed", {}),
            "weather_data": results.get("weather", {}),
            "rag_notes": results.get("rag", []),
            "search_results": results.get("search", {}),
            "feedback": state.get("feedback", "")
        }
        
        response_prompt = f"""
You are an intelligent weather agent. Generate a comprehensive response based on the execution results.

**User Query:** {query}

**Execution Plan:** {plan}

**Execution Results:**
- Tools Used: {', '.join(context['tools_used'])}
- Parsed Query: {context['parsed']}
- Weather Data: {context['weather_data']}
- Knowledge Notes: {context['rag_notes']}
- Search Results: {context['search_results']}
- User Feedback: {context['feedback']}

Generate a response that:
1. Shows the execution plan and reasoning
2. Displays actual execution progress based on what was accomplished
3. Provides intelligent analysis of the weather data
4. Incorporates relevant knowledge and search results
5. Shows learning and adaptation

Format your response with clear sections and emojis. Be conversational but informative.
"""
        
        messages = [
            {"role": "system", "content": "You are an intelligent weather agent. Generate comprehensive, well-structured responses that show your reasoning and execution process."},
            {"role": "user", "content": response_prompt}
        ]
        
        llm_response = self.llm.generate(messages, temperature=0.3, max_tokens=1200)
        return f"{llm_response}\n\n**🤖 Note:** This response was generated by {self.llm.get_provider_info()} analysis of execution results."
    
    def _learn_node(self, state: AgentState) -> AgentState:
        """Learn from interaction using LLM reasoning"""
        
        # Use LLM to generate intelligent learning insights
        learning_insight = self._generate_learning_insight(state)
        rag.add(learning_insight, "learn")
        
        # Store feedback
        if state.get("feedback"):
            with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"query": state["query"], "feedback": state["feedback"]}) + "\n")
        
        return state
    
    def _generate_learning_insight(self, state: AgentState) -> str:
        """Generate learning insights using LLM reasoning"""
        
        results = state["results"]
        query = state["query"]
        
        learning_prompt = f"""
You are an intelligent weather agent learning from user interactions. Generate a concise learning insight.

**User Query:** {query}
**Execution Results:** {results}
**Tools Used:** {', '.join(state['tools'])}

Generate a brief learning insight that captures:
1. What the user was looking for
2. What information was provided
3. Any patterns or insights that could help with future similar queries

Keep it concise and actionable. Format as: "Query: [query] - [insight]"
"""
        
        messages = [
            {"role": "system", "content": "You are a learning system. Generate concise, actionable insights."},
            {"role": "user", "content": learning_prompt}
        ]
        
        learning_insight = self.llm.generate(messages, temperature=0.2, max_tokens=200)
        return learning_insight
    
    def _create_weather_summary(self, location: str, daily: Dict[str, List], units: str) -> str:
        """Create weather summary"""
        dates = daily.get("time", [])
        if not dates:
            return "No forecast data available"
        
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("wind_speed_10m_max", [])
        
        if not all([tmax, tmin]):
            return "Incomplete forecast data"
        
        temp_unit = "°F" if units == "imperial" else "°C"
        wind_unit = "mph" if units == "imperial" else "km/h"
        
        max_temp = max(tmax) if tmax else "?"
        min_temp = min(tmin) if tmin else "?"
        total_rain = sum(p for p in precip if isinstance(p, (int, float))) if precip else 0
        max_wind = max(wind) if wind else "?"
        
        return (f"{location}: {min_temp}-{max_temp}{temp_unit}, "
                f"{total_rain:.1f}mm rain, {max_wind} {wind_unit} max wind")
    
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

# ----------------------- CLI & Tests -----------------------

def run_demo():
    """Run demo queries"""
    agent = WeatherAgent()
    samples = [
        "Weather in Mumbai for 3 days",
        "Any rain alerts for Bengaluru tomorrow?"
    ]
    print(f"\n[LangGraph Weather Agent] Running {len(samples)} demo queries...\n")
    for query in samples:
        result = agent.run(query)
        print(f"Q: {query}\n{result['answer']}\n" + "-"*60)

def run_tests():
    """Run unit tests"""
    import unittest

    class WeatherAgentTests(unittest.TestCase):
        def setUp(self):
            os.environ["AGENT_OFFLINE"] = "1"
        
        def tearDown(self):
            os.environ.pop("AGENT_OFFLINE", None)

        def test_end_to_end_offline(self):
            agent = WeatherAgent()
            result = agent.run("Weather in Pune for 2 days")
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)
            self.assertGreater(len(result["answer"]), 0)
        
        def test_query_parser(self):
            # Test coordinate parsing
            loc, days, units = QueryParser.parse("12.97,77.59 for 2 days in fahrenheit")
            self.assertTrue(re.match(r"^\s*[-+]?\d{1,3}\.\d+\s*,\s*[-+]?\d{1,3}\.\d+\s*$", loc))
            self.assertEqual(days, 2)
            self.assertEqual(units, "imperial")

            # Test city parsing
            loc, days, units = QueryParser.parse("Weather in Mumbai for 3 days")
            self.assertIn("mumbai", loc.lower())
            self.assertEqual(days, 3)
            self.assertEqual(units, "metric")
        
        def test_weather_tool_geocode(self):
            tool = WeatherTool()
            coords = tool._geocode("12.9716,77.5946")
            self.assertIsNotNone(coords)
            self.assertAlmostEqual(coords["latitude"], 12.9716, places=3)
            self.assertAlmostEqual(coords["longitude"], 77.5946, places=3)
        
        def test_rag_functionality(self):
            # Test RAG search
            results = rag.search("timezone uncertainty", limit=2)
            self.assertGreaterEqual(len(results), 1)
    
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WeatherAgentTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="LangGraph Weather Agent")
    parser.add_argument("--query", "-q", help="Weather query to process")
    parser.add_argument("--feedback", "-f", help="Feedback for the agent")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--provider", "-p", choices=["auto", "groq", "openai"], default="auto", 
                       help="LLM provider to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    if args.query:
        try:
            agent = WeatherAgent(llm_provider=args.provider)
            result = agent.run(args.query, feedback=args.feedback)
            print(result["answer"])
        except Exception as e:
            print(f"Error: {e}")
            print("\n💡 Tips:")
            print("- For Groq: Set GROQ_API_KEY in .env file")
            print("- For OpenAI: Set OPENAI_API_KEY in .env file")
            print("- Get Groq API key: https://console.groq.com/keys")
            print("- Get OpenAI API key: https://platform.openai.com/api-keys")
    else:
        run_demo()

if __name__ == "__main__":
    main()
