"""Rag flow that contains the steps to answer a user's question."""
import asyncio

import tenacity
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.indexers.milvus_hybrid import hybrid
from app.rag.prompts import (
    contextulizer_prompt,
    generator_prompt,
    grader_prompt,
    hallucination_prompt,
    regenerator_prompt,
)
from app.static import llm

type ConversationHistory = list[str]
conversation_history: ConversationHistory = []

grader_chain = grader_prompt | llm | StrOutputParser()
compose_chain = generator_prompt | llm | StrOutputParser()
recompose_chain = regenerator_prompt | llm | StrOutputParser()
hallucination_chain = hallucination_prompt | llm | StrOutputParser()
contexulizer_chain = contextulizer_prompt | llm | StrOutputParser()


def format_docs(docs: list[Document]) -> str:
    """
    Formats documents for input into prompt chains.

    Args:
        docs (list[Document]): List of documents.

    Returns:
        str: Formatted document string.

    """
    return "\n".join(
        f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )


async def retrieve(query: str) -> list[Document]:
    """
    Retrieves relevant documents for a given query.

    Args:
        query (str): Query string.

    Returns:
        list[Document]: List of relevant documents.

    """
    return await hybrid.retriever.aget_relevant_documents(query=query)


async def grade_documents(question: str, documents: list[Document]) -> list[Document]:
    """
    Grades documents based on relevance to a question.

    Args:
        question (str): User question.
        documents (list[Document]): List of documents to grade.

    Returns:
        list[Document]: List of approved documents.

    """
    async def grade_document(question: str, document: Document) -> Document | None:
        """Grades a single document for relevance."""
        try:
            res = await grader_chain.ainvoke(
                {"question": question, "document": document.page_content}
            )
            if "yes" in res:
                return document
        except Exception:
            return document

    approved_documents = await asyncio.gather(
        *[
            grade_document(question=question, document=document)
            for document in documents
        ]
    )
    return [doc for doc in approved_documents if doc is not None]


@tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
    wait=tenacity.wait_fixed(1),
)
async def generate_answer(
    question: str, documents: list[Document], history: ConversationHistory
) -> str:
    """
    Generates an answer based on documents and conversation history.

    Args:
        question (str): User question.
        documents (list[Document]): List of relevant documents.
        history (ConversationHistory): User's conversation history.

    Returns:
        str: Generated answer.

    """
    return await compose_chain.ainvoke(
        {"documents": format_docs(documents), "question": question, "history": history}
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
    wait=tenacity.wait_fixed(1),
)
async def regenerate_answer(
    question: str, documents: list[Document], history: ConversationHistory
) -> str:
    """
    Re-generates an answer to reduce hallucinations.

    Args:
        question (str): User question.
        documents (list[Document]): List of relevant documents.
        history (ConversationHistory): User's conversation history.

    Returns:
        str: Re-generated answer.

    """
    return await recompose_chain.ainvoke(
        {"documents": format_docs(documents), "question": question, "history": history}
    )


async def get_hallucination_score(documents: list[Document], generated_answer: str) -> str:
    """
    Evaluates the generated answer for potential hallucination.

    Args:
        documents (list[Document]): List of relevant documents.
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


@tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
    wait=tenacity.wait_fixed(1),
)
async def contextualize_query(question: str, history: ConversationHistory) -> str:
    """
    Contextualizes a query based on recent conversation history.

    Args:
        question (str): User question.
        history (ConversationHistory): User's conversation history.

    Returns:
        str: Contextualized query.

    """
    try:
        query = await contexulizer_chain.ainvoke(
            {"question": question, "history": conversation_history[-4:]}
        )
    except Exception:
        query = question
    return query


async def answer_user(question: str) -> str:
    """
    Processes and generates an answer for a user's question.

    Args:
        question (str): User question.

    Returns:
        str: Generated answer.

    """
    if conversation_history:
        query = await contextualize_query(
            question=question, history=conversation_history[-4:]
        )
    else:
        query = question
    conversation_history.append(f"user question:{query}")
    documents = await retrieve(query)

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
    conversation_history.append(f"AI answer:{answer}")
    return answer
