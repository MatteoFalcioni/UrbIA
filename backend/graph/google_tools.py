from tavily import TavilyClient
import os
from langchain_core.tools import tool

# ---------- Internet Search Tool ----------

@tool
def internet_search(query):
    """Search the internet for information"""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_client.search(query)

# ---------- Google Tools ----------



