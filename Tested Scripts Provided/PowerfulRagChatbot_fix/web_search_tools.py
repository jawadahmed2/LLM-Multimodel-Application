### Search
from config import tavily_api_key
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3, api_key=tavily_api_key)