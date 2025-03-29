from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from interview.interview_agent import InterviewAgent
from interview.web_search_tool import WebSearchTool
from interview.wikipedia_search_tool import WikipediaSearchTool


class InterviewAgentFactory:
    """Factory for creating InterviewAgent and related search tool instances"""

    @staticmethod
    def create_interview_agent(llm: BaseChatModel) -> InterviewAgent:
        """Create an InterviewAgent instance"""
        return InterviewAgent(llm=llm)

    @staticmethod
    def create_web_search_tool(
        llm: BaseChatModel,
        max_results: Optional[int] = 3
    ) -> WebSearchTool:
        """Create a WebSearchTool instance"""
        return WebSearchTool(llm=llm, max_results=max_results)

    @staticmethod
    def create_wikipedia_search_tool(
        llm: BaseChatModel,
        max_docs: Optional[int] = 2
    ) -> WikipediaSearchTool:
        """Create a WikipediaSearchTool instance"""
        return WikipediaSearchTool(llm=llm, max_docs=max_docs)
