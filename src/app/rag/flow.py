"""contains all the functions used in the application llm question/answer flow."""

import asyncio
import json
from collections.abc import Iterable, Sequence

import tenacity
from langchain_core.documents.base import Document as LangchainDocument
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.indexers.milvus_hybrid import milvus
from app.rag.prompts import (
    CONSOLIDATOR_PROMPT,
    CONTEXTUALIZER_PROMPT,
    GENERATOR_PROMPT,
    GRADER_PROMPT,
    HALLUCINATION_PROMPT,
    KEYWORDS_EXTRACTOR_PROMPT,
    REGENERATOR_PROMPT,
)
from app.rag.static import llm

type Keywords = list[str]
type Question = str

grader_chain = GRADER_PROMPT | llm | StrOutputParser()
hallucination_chain = HALLUCINATION_PROMPT | llm | StrOutputParser()
contextualizer_chain = CONTEXTUALIZER_PROMPT | llm | StrOutputParser()
keywords_extractor_chain = KEYWORDS_EXTRACTOR_PROMPT | llm | StrOutputParser()

conversation_history: list[str] = []

compose_chain = GENERATOR_PROMPT | llm | StrOutputParser()
recompose_chain = REGENERATOR_PROMPT | llm | StrOutputParser()
consolidator_chain = CONSOLIDATOR_PROMPT | llm | StrOutputParser()


llm_retry = tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
    wait=tenacity.wait_exponential_jitter(max=10),
)


def format_docs(docs: Sequence[LangchainDocument]) -> str:
    """
    Formats documents for input into prompt chains.

    Args:
        docs (Sequence[LangchainDocument]): List of documents.

    Returns:
        str: Formatted document string.

    """
    return "\n".join(
        f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )


async def grade_documents(
    question: str, documents: Iterable[LangchainDocument]
) -> list[LangchainDocument]:
    """
    Grades documents based on relevance to a question.

    Args:
        question (str): User question.
        documents (Iterable[LangchainDocument]): List of documents to grade.

    Returns:
        list[LangchainDocument]: List of approved documents.

    """
    approved_documents = await asyncio.gather(
        *[grade_document(question=question, content=document) for document in documents]
    )
    return [doc for doc in approved_documents if doc is not None]


async def grade_document(
    question: str, content: LangchainDocument
) -> LangchainDocument | None:
    """
    Grades a single document for relevance.

    the llm answers with `yes` or `no`.
        - `yes` means the document is relevant to the question
        - `no` means it is irrelevant

    Returns:
        LangchainDocument | None: document deemed relevant or None.

    """
    try:
        res = await grader_chain.ainvoke(
            {"question": question, "document": content.page_content}
        )
        if "yes" in res:
            return content
        # document was rejected
        return None
    except Exception:
        # we keep the document when the query fails as it is generally better
        # than missing out on its information because of an exception
        # this is essentially a compromise.
        return content


async def get_hallucination_score(
    documents: Sequence[LangchainDocument], generated_answer: str
) -> str:
    """
    Evaluates the generated answer for potential hallucination.

    Args:
        documents (Sequence[LangchainDocument]): List of relevant documents.
        generated_answer (str): Generated answer.

    Returns:
        str: Hallucination evaluation result.

    """
    try:
        return await hallucination_chain.ainvoke(
            {"documents": format_docs(documents), "generation": generated_answer}
        )
    except Exception:
        return "yes"


async def get_keywords(question: str) -> Keywords:
    """
    get keywords from a question.

    Args:
        question (str): user query

    Returns:
        Keywords: list of keywords

    """
    try:
        str_keywords = await keywords_extractor_chain.ainvoke({"question": question})
        return json.loads(str_keywords)
    except Exception:
        return []


@llm_retry
async def generate_answer(
    question: str, documents: Sequence[LangchainDocument], history: list[str]
) -> str:
    """
    Generates an answer based on documents and conversation history.

    Args:
        question (str): User question.
        documents (Sequence[LangchainDocument]): List of relevant documents.
        history (list[str]): User's conversation history.

    Returns:
        str: Generated answer.

    """
    return await compose_chain.ainvoke(
        {"documents": format_docs(documents), "question": question, "history": history}
    )


@llm_retry
async def regenerate_answer(
    question: str, documents: Sequence[LangchainDocument], history: list[str]
) -> str:
    """
    Re-generates an answer to reduce hallucinations.

    Args:
        question (str): User question.
        documents (Sequence[LangchainDocument]): List of relevant documents.
        history (list[str]): User's conversation history.

    Returns:
        str: Re-generated answer.

    """
    return await recompose_chain.ainvoke(
        {"documents": format_docs(documents), "question": question, "history": history}
    )


async def try_contextualize_query(question: str, history: list[str]) -> str:
    """
    Ty to contextualize a query based on recent conversation history
    otherwise return it as is.

    Args:
        question (str): User question.
        history (list[str]): User's conversation history.

    Returns:
        str: Contextualized query or the original question.

    """
    try:
        return await contextualizer_chain.ainvoke(
            {"question": question, "history": history[-4:]}
        )
    except Exception:
        return question


async def answer_user(question: str) -> str:
    """
    Generated a context aware response to a user query.

    Steps:
        - contextualizes the user question using history
        flow 1:
            - retrieve documents from the vector db using the contextualized query
        flow 2:
            - extract keywords from the contextualized query
            - retrieves documents based on these keywords
        - verifies that the document is related to the query
        - generates an answer
        - ensures no hallucination otherwise retries

    Notes:
        - both aforementioned flows happen consecutively (attempting to run them
        in parallel resulted in issues with Ollama)

    Args:
        question (str): the user's query.

    Returns:
        str: response.

    """
    if conversation_history:
        query = await try_contextualize_query(
            question=question, history=conversation_history[-4:]
        )
    else:
        query = question
    conversation_history.append(f"user question:{query}")

    query_answer = await answer_using_query(question)
    keyword_answer = await answer_using_keywords(question)
    answer = await consolidate_answers(
        user_query=query, answers=[query_answer, keyword_answer]
    )
    conversation_history.append(f"AI answer:{answer}")
    return answer


async def answer_using_documents(
    query: str, documents: Iterable[LangchainDocument]
) -> str:
    """
    Processes and generates an answer for a user's question.

    Args:
        query (str): User question.
        documents (list[LangchainDocument]) : list of retrieved documents

    Returns:
        str: Generated answer.

    """
    chosen_documents = await grade_documents(question=query, documents=documents)
    if len(chosen_documents) == 0:
        generated_answer = (
            "I apologize but I could not find an answer to your question."
        )
    try:
        generated_answer = await generate_answer(
            question=query,
            documents=chosen_documents,
            history=conversation_history[-4:],
        )
    except Exception:
        generated_answer = (
            "I apologize but I could not find an answer to your question."
        )
    hallucination_score = await get_hallucination_score(
        documents=chosen_documents, generated_answer=generated_answer
    )
    if "no" in hallucination_score:
        try:
            answer = await regenerate_answer(
                question=query,
                documents=chosen_documents,
                history=conversation_history[-4:],
            )
        except Exception:
            return generated_answer
    else:
        answer = generated_answer
    return answer


async def consolidate_answers(user_query: str, answers: list[str]) -> str:
    """
    function  to consolidate answers into one answer.

    Args:
        user_query (str): user question
        answers (list[str]): list of llm answers

    Returns:
        str: answer to the query

    """
    try:
        answer = await consolidator_chain.ainvoke(
            {"question": user_query, "answers": answers}
        )
    except Exception:
        answer = answers[0]
    return answer


async def answer_using_keywords(question: str) -> str:
    """
    use keywords to retrieve documents and answer the user query.

    Args:
        question (str): user query

    Returns:
        str : the llm answer.

    """
    keywords = await get_keywords(question)
    if not keywords:
        return ""
    documents = await milvus.batch_retrieve(keywords)
    return await answer_using_documents(query=question, documents=documents)


async def answer_using_query(question: str) -> str:
    """
    use the query to directly retrieve documents and answer the user.

    Args:
        question (str): user query

    Returns:
        str : the llm answer.

    """
    documents = await milvus.retrieve(question)
    return await answer_using_documents(query=question, documents=documents)
