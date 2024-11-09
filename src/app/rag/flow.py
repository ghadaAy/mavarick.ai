import asyncio
import json
from collections.abc import Callable, Iterable
from typing import Literal

import tenacity
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.indexers.milvus_hybrid import hybrid
from app.rag.prompts import (
    consolidator_prompt,
    contextulizer_prompt,
    generator_prompt,
    grader_prompt,
    hallucination_prompt,
    keywords_extractor_prompt,
    regenerator_prompt,
)
from app.static import llm

type ConversationHistory = list[str]
type Keywords =list[str]
type Question=str
type ListDocuments=Iterable[Document]

grader_chain = grader_prompt | llm | StrOutputParser()
hallucination_chain = hallucination_prompt | llm | StrOutputParser()
contexulizer_chain = contextulizer_prompt | llm | StrOutputParser()
keywords_extractor_chain = keywords_extractor_prompt | llm | StrOutputParser()

def format_docs(docs: ListDocuments) -> str:
    """
    Formats documents for input into prompt chains.

    Args:
        docs (ListDocuments): List of documents.

    Returns:
        str: Formatted document string.

    """
    return "\n".join(
        f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )

async def grade_documents(question: Question, documents: ListDocuments) -> ListDocuments:
    """
    Grades documents based on relevance to a question.

    Args:
        question (str): User question.
        documents (ListDocuments): List of documents to grade.

    Returns:
        ListDocuments: List of approved documents.

    """
    async def grade_document(question: Question, document: Document) -> Document | None:
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

async def get_hallucination_score(documents: ListDocuments, generated_answer: str) -> str:
    """
    Evaluates the generated answer for potential hallucination.

    Args:
        documents (ListDocuments): List of relevant documents.
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

async def get_keywords(question: Question)->Keywords:
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
conversation_history: ConversationHistory = []

compose_chain = generator_prompt | llm | StrOutputParser()
recompose_chain = regenerator_prompt | llm | StrOutputParser()
consolidator_chain = consolidator_prompt | llm | StrOutputParser()


@tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
    wait=tenacity.wait_fixed(1),
)
async def generate_answer(
    question: Question, documents: ListDocuments, history: ConversationHistory
) -> str:
    """
    Generates an answer based on documents and conversation history.

    Args:
        question (str): User question.
        documents (ListDocuments): List of relevant documents.
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
    question: Question, documents: ListDocuments, history: ConversationHistory
) -> str:
    """
    Re-generates an answer to reduce hallucinations.

    Args:
        question (str): User question.
        documents (ListDocuments): List of relevant documents.
        history (ConversationHistory): User's conversation history.

    Returns:
        str: Re-generated answer.

    """
    return await recompose_chain.ainvoke(
        {"documents": format_docs(documents), "question": question, "history": history}
    )




@tenacity.retry(
    stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
    wait=tenacity.wait_fixed(1),
)
async def contextualize_query(question: Question, history: ConversationHistory) -> str:
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

async def answer_user(question: Question) -> str:
    if conversation_history:
        query = await contextualize_query(
            question=question, history=conversation_history[-4:]
        )
    else:
        query = question
    conversation_history.append(f"user question:{query}")

    query_answer =  await answer_using_query(question)
    keyword_answer = await answer_using_keywords(question)
    answer = await consolidate_answers(user_query=query, answers=[query_answer,  keyword_answer])
    conversation_history.append(f"AI answer:{answer}")
    return answer

async def answer_using_documents(query: Question, documents:ListDocuments|list[str]) -> str:
    """
    Processes and generates an answer for a user's question.

    Args:
        question (str): User question.
        documents (list[Document]) : list of retrieved documents

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



async def consolidate_answers(user_query:Question,answers:list[str]) -> str:
    """
    function  to consolidate answers into one answer.

    Args:
        user_query (Question): user question
        answers (list[str]): list of llm answers

    Returns:
        str: answer to the query

    """
    try:
        answer = await consolidator_chain.ainvoke({"question":user_query, "answers":answers})
    except Exception:
        answer= answers[0]
    return answer

async def answer_using_keywords(question:Question) -> str:
    """
    use keywords to retrieve documents and answer the user query.

    Args:
        question (str): user query
    Returns
        str : the llm answer.

    """
    keywords = await get_keywords(question)
    if not keywords:
        return ""
    documents = await hybrid.batch_retrieve(keywords)
    return await answer_using_documents(query=question, documents=documents)

async def answer_using_query(question:Question)->str:
    """
    _summary_.

    Args:
        question (Question): _description_

    Returns:
        _type_: _description_

    """
    documents = await hybrid.retrieve(question)
    return await answer_using_documents(query=question, documents=documents)
