from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")


@mcp.tool()
def add(a:int, b:int) -> int:
    """
        Given two integer values a and b, this returns sum of them
        Args:
            a : integer
            b : integer
        return 
            integer
    """
    return a+b


@mcp.tool()
def multiply(a:int, b:int) -> int:
    """
        Given two integer values a and b, this returns multiply of them
        Args:
            a : integer
            b : integer
        return 
            integer
    """
    return a*b

# The transspot='stdio' arguments tell the server to :
# Use standard input/output (stdin, stdout) to receive and respond to the tool function calls.

if __name__=="__main__":
    mcp.run(transport="stdio")