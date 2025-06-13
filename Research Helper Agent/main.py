from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from firecrawl import FirecrawlApp, ScrapeOptions
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    topic: str
    report: str
    summary: str

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

summarize_research_prompt = """
"""

system_prompt = """
"""

enhancement_prompt = """
"""

api_url="http://localhost:3002/v1/"

@tool
def web_search(topic: str, limit: int, time_frame: str) -> list:
    """
    This tool will do web search about any given topics,
    it will do websearch using firecrawl search data,
    it will return list of websearch result

    Args:
        topic: Name or context of topic to research about max 100 chars
        limit: result limit
        time_frame: freshness // qdr:h - Past hour, qdr:d - Past 24 hours, qdr:w - Past week, qdr:m - Past month, qdr:y - Past year

    output:
        list: output would be in list 
            example output:
            [
                {
                    "url": "https://www.octoparse.com/",
                    "title": "Octoparse: Web Scraping Tool & Free Web Crawlers",
                    "description": "  Easy Web Scraping for Anyone. Octoparse is your no-coding solution for web scraping to turn pages into structured data within clicks. Start a free trial  "
                },
                {
                    "url": "https://www.scraperapi.com/web-scraping/tools/",
                    "title": "14 Best Web Scraping Tools In 2025 (Pros, Cons, Pricing)",
                    "description": "  cURL. Collect data at scale from your terminal. Python. Collect and analyze data with a single language. NodeJS. Build robust scrapers the simple way. PHP.  "
                }
            ]
    """
    app = FirecrawlApp(api_url=api_url + "/search")
    search_result = app.search(
        topic, 
        limit=limit,
        scrape_options=ScrapeOptions(formats=["markdown"]),
        tbs=time_frame
    )

    return search_result.data

@tool
def submit_web_crawl_request(url: str, limit: int) -> str:
    """
    Use this tool for 
        URL Analysis: Scans sitemap and crawls website to identify links
        Traversal: Recursively follows links to find all subpages
        Scraping: Extracts content from each page, handling JS and rate limits

        Args:
            url: URL of website to crawl
            limit: result limit

        Output: will return requestID of web crawl
            example_output: 288a347f-9299-4d9b-a29a-97a04442ca58
    """
    app = FirecrawlApp(api_url=api_url + "/crawl")
    crawl_result = app.crawl_url(
        url, 
        limit=limit, 
        scrape_options=ScrapeOptions(formats=['markdown']),
    )
    if crawl_result['status'] == 'error':
        return crawl_result['error']
    return crawl_result['id']

@tool
def get_web_crawl_result(firecrawl_request_id: str) -> list:
    """
    Use this tool to get the result of web crawl request

    Args:
        firecrawl_request_id: requestID of web crawl (example: 288a347f-9299-4d9b-a29a-97a04442ca58)

    Output: will return the result of web crawl this will be in list
        Output_Example: [
            {
                "markdown": "information in MD Formate",
                "metadata": {
                    "description": "Firecrawl Rust SDK is a library to help you easily scrape and crawl websites, and output the data in a format ready for use with language models (LLMs)."
                }
            }
        ]
    """
    app = FirecrawlApp(api_url=api_url + "/crawl")
    crawl_status = app.check_crawl_status(firecrawl_request_id)

    data = crawl_status.data
    for _ in data:
        [_]['metedata'] = {k: v for k, v in [_]['metedata'].items() if k == "description"}

    return data


@tool
def summarize_research(research_data: str) -> str:
    """
    After successfully getting report topic from research tool,
    this tool will help it summuarize the entire report and explain out in simpler term for user.

    Args:
        research_data: Entire report, to generate the summary
    """
    local_llm_node = ChatOpenAI(model="gpt-4o-mini")

    user_message = HumanMessage(content=research_data)
    msg = [summarize_research_prompt] + ['\n\n' + user_message]
    prompt = SystemMessage(content=msg)
    response = local_llm_node.invoke(prompt)

    return response.content


def save(paper: str) -> str:
    """Determine if we should save or discard the report."""

    save = input(" 🧠 AI : Do you want to save the report? (Y/n)").strip().lower()

    if save == 'y':
        pass
    else:
        pass

    return 'success'

@tool
def output_display(state: AgentState) -> AgentState:
    """
    After getting summary of report it will display the summary to user

    Args:
        state: entire node state 
    """
    
    save(state['report'])
    save(state['summary'])

    return state


tools = [web_search, submit_web_crawl_request, get_web_crawl_result, summarize_research, output_display]
llm.bind_tools(tools=tools)

def init_node(state: AgentState) -> AgentState:
    """Initially Ask about search topic"""

    # research_topic = input(" 🧠 AI : Please inseart the topic you want to research about : ").strip().lower()
    research_topic = 'AI agent'

    if research_topic == 'exit' or research_topic == 'quit' or research_topic == "":
        import sys
        sys.exit(0)

    state['topic'] = research_topic

    return state


def prompt_enhancer(state: AgentState) -> AgentState:
    """This AI node is used for prompt enhancement"""

    llm_node = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)

    user_message = HumanMessage(content=state['topic'])
    prompt = [enhancement_prompt] + ['\n\n' + user_message]
    response = llm_node.invoke(prompt)

    state['topic'] = response.content

    return state


def agent(state: AgentState) -> AgentState:
    """AI Agent to deeply run research about"""
    
    message = [system_prompt] + ["\n\n Research Topic:\n" + state['topic']]

    response = llm.invoke(message)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return state


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("init_node", init_node)
graph.add_node("prompt_enhancer", prompt_enhancer)
graph.add_node("agent", agent)

graph.add_edge(START, "init_node")
graph.add_edge("init_node", "prompt_enhancer")
graph.add_edge("prompt_enhancer", "agent")
graph.add_edge('agent', END)

app = graph.compile()

def run_researcher_agent():
    print("\n ===== RESEARCHER =====")
    state = {"topic": '', "report": '', 'summary': ''}
    app.invoke(state)
    print("\n ===== RESEARCHER FINISHED =====")

if __name__ == "__main__":
    run_researcher_agent()