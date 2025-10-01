"""Tools for weather agent"""

import os
import re
import json
import requests
from typing import Any, Dict, List, Optional
from tool_base import BaseTool, ToolResult
from config import config


class WeatherTool(BaseTool):
    """Weather tool using Open-Meteo API"""
    
    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        super().__init__("weather")
    
    def _execute(self, location: str, days: int = None, units: str = "metric") -> Dict[str, Any]:
        """Get weather forecast"""
        days = days or config.default_forecast_days
        days = max(config.min_forecast_days, min(config.max_forecast_days, days))
        
        # Get coordinates
        coords = self._geocode(location)
        if not coords:
            raise ValueError(f"Could not find location: {location}")
        
        # Get forecast
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "sunrise", "sunset"],
            "forecast_days": days,
            "temperature_unit": "fahrenheit" if units == "imperial" else "celsius",
            "wind_speed_unit": "mph" if units == "imperial" else "kmh",
        }
        
        response = requests.get(self.FORECAST_URL, params=params, timeout=config.weather_timeout)
        response.raise_for_status()
        data = response.json()
        data["resolved"] = coords
        return data
    
    def _geocode(self, location: str) -> Optional[Dict[str, float]]:
        """Get coordinates for location"""
        # Handle coordinates directly
        coords_match = re.match(r"^\s*([-+]?\d{1,3}\.\d+)\s*,\s*([-+]?\d{1,3}\.\d+)\s*", location)
        if coords_match:
            return {
                "latitude": float(coords_match.group(1)),
                "longitude": float(coords_match.group(2)),
                "name": location
            }
        
        # Use geocoding API
        params = {"name": location, "count": 1}
        try:
            response = requests.get(self.GEO_URL, params=params, timeout=config.weather_timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"],
                    "name": result["name"]
                }
        except Exception:
            pass
        return None
    
    def _get_meta(self, **kwargs) -> Dict[str, Any]:
        """Get metadata for weather tool execution"""
        return {
            "tool": self.name,
            "location": kwargs.get("location"),
            "days": kwargs.get("days", config.default_forecast_days),
            "units": kwargs.get("units", "metric")
        }


class SearchTool(BaseTool):
    """Search tool using SerpAPI"""
    
    def __init__(self):
        super().__init__("search")
    
    def _execute(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search for weather-related information"""
        limit = limit or config.search_limit
        
        if not self.is_available():
            raise ValueError("SERPAPI_API_KEY not configured")
        
        import serpapi
        search = serpapi.GoogleSearch({
            "q": f"{query} weather alert news",
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": limit
        })
        
        results = search.get_dict()
        return results.get("organic_results", [])
    
    def is_available(self) -> bool:
        """Check if search tool is available"""
        return bool(os.getenv("SERPAPI_API_KEY"))
    
    def _get_meta(self, **kwargs) -> Dict[str, Any]:
        """Get metadata for search tool execution"""
        return {
            "tool": self.name,
            "query": kwargs.get("query"),
            "limit": kwargs.get("limit", config.search_limit)
        }


class SimpleRAG(BaseTool):
    """RAG system for knowledge management"""
    
    def __init__(self):
        super().__init__("rag")
        self._ensure_seeded()
    
    def _ensure_seeded(self):
        """Initialize with basic knowledge if file doesn't exist"""
        if not config.rag_path.exists():
            self.add("Use local timezone and mention uncertainty after day 3.", "seed")
            self.add("Open-Meteo provides daily tmax/tmin, precip sum, wind gusts, sunrise/sunset.", "seed")
    
    def add(self, text: str, source: str = "user"):
        """Add knowledge to RAG"""
        with open(config.rag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "source": source}) + "\n")
    
    def _execute(self, query: str, limit: int = None) -> List[str]:
        """Search for relevant knowledge"""
        limit = limit or config.rag_limit
        
        if not config.rag_path.exists():
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for line in config.rag_path.read_text(encoding="utf-8").splitlines():
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
    
    def _get_meta(self, **kwargs) -> Dict[str, Any]:
        """Get metadata for RAG execution"""
        return {
            "tool": self.name,
            "query": kwargs.get("query"),
            "limit": kwargs.get("limit", config.rag_limit)
        }
