"""Configuration management"""

import os
import pathlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for the weather agent"""
    
    # LLM Configuration
    llm_provider: str = "auto"
    
    # Weather API Configuration
    max_forecast_days: int = 7
    min_forecast_days: int = 1
    default_forecast_days: int = 3
    
    # RAG Configuration
    rag_limit: int = 3
    rag_file: str = "notes.jsonl"
    
    # Search Configuration
    search_limit: int = 3
    
    # File Paths
    base_dir: pathlib.Path = pathlib.Path.cwd()
    memory_dir: Optional[pathlib.Path] = None
    rag_path: Optional[pathlib.Path] = None
    feedback_path: Optional[pathlib.Path] = None
    
    # API Timeouts
    weather_timeout: int = 12
    search_timeout: int = 10
    
    def __post_init__(self):
        """Initialize derived paths"""
        if self.memory_dir is None:
            self.memory_dir = self.base_dir / "memory"
        
        if self.rag_path is None:
            self.rag_path = self.memory_dir / self.rag_file
        
        if self.feedback_path is None:
            self.feedback_path = self.memory_dir / "feedback.jsonl"
        
        # Ensure memory directory exists
        self.memory_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment"""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "auto"),
            max_forecast_days=int(os.getenv("MAX_FORECAST_DAYS", "7")),
            min_forecast_days=int(os.getenv("MIN_FORECAST_DAYS", "1")),
            default_forecast_days=int(os.getenv("DEFAULT_FORECAST_DAYS", "3")),
            rag_limit=int(os.getenv("RAG_LIMIT", "3")),
            search_limit=int(os.getenv("SEARCH_LIMIT", "3")),
            weather_timeout=int(os.getenv("WEATHER_TIMEOUT", "12")),
            search_timeout=int(os.getenv("SEARCH_TIMEOUT", "10"))
        )
    
    def validate(self) -> None:
        """Validate configuration"""
        if self.max_forecast_days < self.min_forecast_days:
            raise ValueError("max_forecast_days must be >= min_forecast_days")
        
        if self.default_forecast_days < self.min_forecast_days:
            raise ValueError("default_forecast_days must be >= min_forecast_days")
        
        if self.default_forecast_days > self.max_forecast_days:
            raise ValueError("default_forecast_days must be <= max_forecast_days")
        
        if self.rag_limit <= 0:
            raise ValueError("rag_limit must be > 0")
        
        if self.search_limit <= 0:
            raise ValueError("search_limit must be > 0")


# Global configuration instance
config = Config.from_env()
config.validate()
