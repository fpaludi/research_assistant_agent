from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from interview.io_models import InterviewState, SearchQuery


class BaseSearchTool(ABC):
    _SEARCH_INSTRUCTIONS = """You will be given a conversation between an analyst and an expert.

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured search query"""

    def __init__(self, llm: BaseChatModel):
        """Initialize the base search tool with a language model."""
        self.llm = llm
        self.structured_llm = llm.with_structured_output(SearchQuery)

    def _generate_search_query(self, state: InterviewState) -> SearchQuery:
        """Generate a search query from the conversation state."""
        result = self.structured_llm.invoke([
            SystemMessage(content=self._SEARCH_INSTRUCTIONS)
        ] + state['messages'])
        # Since we're using structured output, result should already be SearchQuery
        return result

    @abstractmethod
    def search(self, state: InterviewState) -> dict:
        """Execute the search and return results."""
        pass

    @staticmethod
    def format_documents(docs: list, template: str) -> str:
        """Format a list of documents into a string using the provided template."""
        return "\n\n---\n\n".join(template.format(**doc) for doc in docs)
