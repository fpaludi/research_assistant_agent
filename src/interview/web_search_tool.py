from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from interview.base_search_tool import BaseSearchTool
from interview.io_models import InterviewState


class WebSearchTool(BaseSearchTool):
    def __init__(self, llm: BaseChatModel, max_results: int = 3):
        """Initialize the web search tool with a language model and search config."""
        super().__init__(llm)
        self.tavily_search = TavilySearchResults(max_results=max_results)

    def search(self, state: InterviewState) -> dict:
        """Retrieve docs from web search"""
        # Generate search query
        search_query = self._generate_search_query(state)

        # Execute search
        search_docs = self.tavily_search.invoke(search_query.search_query)

        # Format results
        formatted_search_docs = self.format_documents(
            search_docs,
            '<Document href="{url}"/>\n{content}\n</Document>'
        )

        return {"context": [formatted_search_docs]}
