"""Base tool class"""

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Tool result format"""
    name: str
    ok: bool
    data: Any
    meta: Optional[Dict[str, Any]] = None


class BaseTool:
    """Base class for all tools"""
    
    def __init__(self, name: str):
        self.name = name
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        try:
            data = self._execute(**kwargs)
            return ToolResult(
                name=self.name,
                ok=True,
                data=data,
                meta=self._get_meta(**kwargs)
            )
        except Exception as e:
            return ToolResult(
                name=self.name,
                ok=False,
                data=str(e),
                meta=self._get_meta(**kwargs)
            )
    
    def _execute(self, **kwargs) -> Any:
        """Tool-specific execution logic"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _execute method")
    
    def _get_meta(self, **kwargs) -> Dict[str, Any]:
        """Get metadata for the tool execution"""
        return {"tool": self.name}
    
    def is_available(self) -> bool:
        """Check if the tool is available for use"""
        return True
