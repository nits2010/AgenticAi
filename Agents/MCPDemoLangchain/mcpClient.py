from dotenv import load_dotenv
load_dotenv()


from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent # this is require to work with mcp
from langchain_groq import ChatGroq
import asyncio,os

async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        raise ValueError("GROQ_API_KEY environment variable not set")
    os.environ["GROQ_API_KEY"] = api_key
    
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathMcpServer.py"],  # Use absolute path if needed
                "transport": "stdio"
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http"
            }
            # Add more servers as needed
        }
    ) # type: ignore
    
    #get all tools/mcp server available, defined by the client
    tools = await client.get_tools()
    
    #setup the LLM model 
    model=ChatGroq(model="qwen-qwq-32b")
    
    #create reach agent with model and tools
    agent=create_react_agent(
        model, tools
    )
    
    math_response = await agent.ainvoke(
       {"messages":[{"role":"user", "content":"What is (3*5)+3"}]}
    )
    print("Math response",{math_response['messages'][-1].content})
    
    weather_response = await agent.ainvoke(
       {"messages":[{"role":"user", "content":"What is weather in india"}]}
    )
    print("Weather response",{weather_response['messages'][-1].content})
           


if __name__ == "__main__":
    asyncio.run(main())