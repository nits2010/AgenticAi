from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str) -> str:
    """
        Get the weather of the given location
        
    """
    #for now just adding constant value
    return "It's rainy"


if __name__=="__main__":
    mcp.run(transport="streamable-http")