"""All prompts used in the rag flow."""

from langchain_core.prompts import ChatPromptTemplate

GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    if you have any doubts, return 'yes'
    you must return only yes or no.
    if you think the answer can be used to answer as it contains partial information return yes.
    be smart and attentive.
    example output:
    no
    """

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", GRADER_SYSTEM_PROMPT),
        (
            "human",
            "Retrieved documents: \n\n {document} \n\n User question: {question}",
        ),
    ]
)

GENERATOR_SYSTEM_PROMPT = """You are a RAG agent, your job is to look for any information inside of the context that can be used in answering the question.
Be thorough, focus, read well, take your time we're not in a hurry
You do not have the right to miss any part of the answer so focus
there is no need for the parts of the answer to be formatted the same way to be included in the answer,
also be careful and only reference the text
please answer correctly and do not hallucinate.
It is very important that you look for the answer to any part of the question inside each piece of text.
Look closely into each paragraph for a potential answer
please focus and read the docs well before answering
if a potential answer is provided in multiple locations, present all of them.
use the conversation history to  understand the user question better

IMPORTANT:some contexts in the middle might contain an answer, look thoroughly though them
never return Some of the projects include always return all of the details.
do not make assumptions and only respond by what is available in  the context
If you cannot answer return "I apologize but I could not find an answer to your question."
"""

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", GENERATOR_SYSTEM_PROMPT),
        (
            "human",
            "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>, \n\n conversation history <history>{history}</history>",
        ),
    ]
)

HALLUCINATION_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
    if you have any doubts, return 'yes'
    you must return only yes or no.
    example output:
    no"""
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", HALLUCINATION_SYSTEM_PROMPT),
        (
            "human",
            "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>",
        ),
    ]
)

REGENERATOR_SYSTEM_PROMPT = """You are a RAG agent, your job is to look for any information inside of the context that can be used in answering the question.
Be thorough, focus, read well, take your time we're not in a hurry
You do not have the right to miss any part of the answer so focus
there is no need for the parts of the answer to be formatted the same way to be included in the answer,
also be careful and only reference the text
please answer correctly and do not hallucinate.
It is very important that you look for the answer to any part of the question inside each piece of text.
Look closely into each paragraph for a potential answer
please focus and read the docs well before answering
if a potential answer is provided in multiple locations, present all of them.
IMPORTANT:some contexts in the middle might contain an answer, look thoroughly though them
never return Some of the projects include always return all of the details.
do not make assumptions and only respond by what is available in  the context.
use the conversation history to  understand the user question better
If you cannot answer return "I apologize but I could not find an answer to your question."
you have hallucinated your answer beforehand! make sure to only use the retrieved documents.
"""
REGENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", REGENERATOR_SYSTEM_PROMPT),
        (
            "human",
            "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n conversation history <history>{history}</history>",
        ),
    ]
)

CONTEXTUALIZER_SYSTEM_PROMPT = """

you are a question contextualizer, Your only task is to give context to the user's query and create a standalone query.
focus, think and take your time, follow these steps:
Your job does not involve answering the user.

\nGiven a chat history and the latest user query \
which might reference context in the chat history, formulate a standalone query \
which can be understood without the chat history.
Do NOT answer the user, \
just reformulate their query if needed and otherwise return it as is.
\n
Please keep the meaning of the query intact and do not alter it to become something completely different from
the user's intent.\n


Examples:
User Question: 'What's a dream?'
AI Response: 'What is the definition of a dream?'
The goal is to have the highest score possible.

User Statement: 'You are nice.'
AI Response: 'You are nice.'

User Statement: 'The weather is bad today.'
AI Response: 'The weather is bad today.'

"""
CONTEXTUALIZER_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", CONTEXTUALIZER_SYSTEM_PROMPT),
        (
            "human",
            "user query: \n\n <query>{question}</query> \n\n history: <history>{history}</history>",
        ),
    ]
)

KEYWORDS_EXTRACTOR_SYSTEM_PROMPT = """Extract only the essential keywords from the following query and return them in a list format. Ensure the output is a JSON-parsable list of keywords.

Example output (only as a format guide, not actual content): ["keyword1", "keyword2"]
please only return the list just like this ["keyword",  "keyword1"].
"""
KEYWORDS_EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", KEYWORDS_EXTRACTOR_SYSTEM_PROMPT),
        (
            "human",
            "user query: \n\n <query>{question}</query> ",
        ),
    ]
)


CONSOLIDATOR_SYSTEM_PROMPT = """Given a user query and a list of answers, consolidate the answers into a single, cohesive response that directly addresses the query. Limit the response to a maximum of 4 sentences, ensuring clarity and conciseness. If one of the answers is "Sorry, I could not answer the question," ignore it and use the other answer. If all answers are "Sorry, I could not answer the question," return "Sorry, I could not answer the question."
"""
CONSOLIDATOR_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", CONSOLIDATOR_SYSTEM_PROMPT),
        (
            "human",
            "user query: \n\n <query>{question}</query> \n\n <answers>{answers}</answers>",
        ),
    ]
)
