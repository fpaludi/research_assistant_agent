from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Send
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    get_buffer_string
)
from interview.io_models import InterviewState
from graph.io_models import ResearchGraphState


class InterviewAgent:
    _QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

    Your goal is boil down to interesting and specific insights related to your topic.

    1. Interesting: Insights that people will find surprising or non-obvious.

    2. Specific: Insights that avoid generalities and include specific examples from the expert.

    Here is your topic of focus and set of goals: {goals}

    Begin by introducing yourself using a name that fits your persona, and then ask your question.

    Continue to ask questions to drill down and refine your understanding of the topic.

    When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

    Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


    _ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.

    Here is analyst area of focus: {goals}.

    You goal is to answer a question posed by the interviewer.

    To answer question, use this context:

    {context}

    When answering questions, follow these guidelines:

    1. Use only the information provided in the context.

    2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

    3. The context contain sources at the topic of each individual document.

    4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1].

    5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc

    6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list:

    [1] assistant/docs/llama3_1.pdf, page 7

    And skip the addition of the brackets as well as the Document source preamble in your citation."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the interview agent with a language model."""
        self.llm = llm

    def initiate_all_interviews(self, state: ResearchGraphState):
        """ This is the "map" step where we run each interview sub-graph using Send API """

        topic = state["topic"]
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
                }
            )
            for analyst in state["analysts"]
        ]

    def generate_question(self, state: InterviewState) -> dict:
        """Node to generate a question"""

        # Get state
        analyst = state["analyst"]
        messages = state["messages"]

        # Generate question
        system_message = self._QUESTION_INSTRUCTIONS.format(goals=analyst.persona)
        question = self.llm.invoke([SystemMessage(content=system_message)] + messages)

        # Write messages to state
        return {"messages": [question]}

    def generate_answer(self, state: InterviewState):
        """ Node to answer a question """

        # Get state
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]

        # Answer question
        system_message = self._ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
        answer = self.llm.invoke([SystemMessage(content=system_message)] + messages)

        # Name the message as coming from the expert
        answer.name = "expert"

        # Append it to state
        return {"messages": [answer]}

    def save_interview(self, state: InterviewState):
        """Save interviews """

        # Get messages
        messages = state["messages"]

        # Convert interview to a string
        interview = get_buffer_string(messages)

        # Save to interviews key
        return {"interview": interview}

    def route_messages(
        self,
        state: InterviewState,
        name: str = "expert"
    ):
        """ Route between question and answer """

        # Get messages
        messages = state["messages"]
        max_num_turns = state.get('max_num_turns',2)

        # Check the number of expert answers
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )

        # End if expert has answered more than the max turns
        if num_responses >= max_num_turns:
            return 'save_interview'

        # This router is run after each question - answer pair
        # Get the last question asked to check if it signals the end of discussion
        last_question = messages[-2]

        next_node = "ask_question"
        if "Thank you so much for your help" in last_question.content:
            next_node = 'save_interview'
        return next_node
