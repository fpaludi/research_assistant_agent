from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langchain_core.messages import HumanMessage

from graph.io_models import ResearchGraphState
from interview.io_models import InterviewState
from llm.model_factory import LLMFactory
from analyst.analyst_factory import AnalystAgentFactory
from interview.interview_factory import InterviewAgentFactory
from report.report_factory import ReportAgentFactory

# Get LLM
llm_factory = LLMFactory()
llm = llm_factory.create("gpt-o4")

# Create agents using factories
analyst_agent = AnalystAgentFactory.create(llm, "initiate_interviews")
interview_agent = InterviewAgentFactory.create_interview_agent(llm)
web_search = InterviewAgentFactory.create_web_search_tool(llm)
wiki_search = InterviewAgentFactory.create_wikipedia_search_tool(llm)
report_agent = ReportAgentFactory.create(llm)

## Interview Graph
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", interview_agent.generate_question)
interview_builder.add_node("search_web", web_search.search)
interview_builder.add_node("search_wikipedia", wiki_search.search)
interview_builder.add_node("answer_question", interview_agent.generate_answer)
interview_builder.add_node("save_interview", interview_agent.save_interview)
interview_builder.add_node("write_section", report_agent.write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question",
    interview_agent.route_messages,
    ['ask_question', 'save_interview']
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview
# memory_interview = MemorySaver()
# interview_graph = interview_builder.compile(
#     checkpointer=memory_interview
# ).with_config(run_name="Conduct Interviews")


# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", analyst_agent.create_analysts)
builder.add_node("human_feedback", analyst_agent.human_feedback)
builder.add_node("initiate_interviews", interview_agent.initiate_all_interviews)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", report_agent.write_report)
builder.add_node("write_introduction", report_agent.write_introduction)
builder.add_node("write_conclusion", report_agent.write_conclusion)
builder.add_node("finalize_report", report_agent.finalize_report)

def initiate_all_interviews(state: ResearchGraphState):

    """ Conditional edge to initiate all interviews via Send() API or return to create_analysts """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback','approve')
    if human_analyst_feedback.lower() != 'approve':
        # Return to create_analysts
        return "create_analysts"

    else:
    # Otherwise kick off interviews in parallel via Send() API
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])

builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# memory = MemorySaver()
graph = builder.compile(
    interrupt_before=['human_feedback'],
    # checkpointer=memory
)

# graph.get_graph(xray=1).draw_png("agent_graph.png")
