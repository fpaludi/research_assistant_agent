from langchain_community.document_loaders import WikipediaLoader
from langchain_core.language_models.chat_models import BaseChatModel
from interview.base_search_tool import BaseSearchTool
from interview.io_models import InterviewState


class WikipediaSearchTool(BaseSearchTool):
    def __init__(self, llm: BaseChatModel, max_docs: int = 2):
        """Initialize the wikipedia search tool with a language model and search config."""
        super().__init__(llm)
        self.max_docs = max_docs

    def search(self, state: InterviewState) -> dict:
        """Retrieve docs from wikipedia"""
        # Generate search query
        search_query = self._generate_search_query(state)

        # Search
        search_docs = WikipediaLoader(
            query=search_query.search_query,
            load_max_docs=self.max_docs
        ).load()

        # Format results with metadata
        docs = [{
            'source': doc.metadata["source"],
            'page': doc.metadata.get("page", ""),
            'content': doc.page_content
        } for doc in search_docs]

        formatted_search_docs = self.format_documents(
            docs,
            '<Document source="{source}" page="{page}"/>\n{content}\n</Document>'
        )

        return {"context": [formatted_search_docs]}