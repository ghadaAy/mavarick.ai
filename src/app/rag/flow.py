import asyncio

import tenacity
from langchain_core.documents.base import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

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
conversation_history:ConversationHistory=[]

grader_chain=grader_prompt | llm | StrOutputParser()

compose_chain = generator_prompt | llm | StrOutputParser()
recompose_chain = regenerator_prompt | llm | StrOutputParser()

hallucination_chain=hallucination_prompt | llm | StrOutputParser()
contexulizer_chain=contextulizer_prompt | llm | StrOutputParser()

def format_docs(docs:list[Document]) -> str:
    return "\n".join(f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))


async def retrieve(query:str) -> list[Document]:
    return await hybrid.retriever.aget_relevant_documents(query=query)


async def grade_documents(question:str, documents:list[Document])->list[Document]:
    async def grade_document(question:str, document:Document) -> Document | None:
        try:
            res = await grader_chain.ainvoke({"question": question, "document": document.page_content})
            if "yes" in res:
                return document
        except Exception:
            return document
    approved_documents = await asyncio.gather(*[grade_document(question=question,document=document) for document in documents])
    return [doc for doc in approved_documents if doc is not None]

@tenacity.retry(
        stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
        wait=tenacity.wait_fixed(1),
    )
async def generate_answer(question:str,documents:list[Document], history:ConversationHistory) -> str:
    return await compose_chain.ainvoke({"documents":format_docs(documents), "question": question, "history":history})

@tenacity.retry(
        stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
        wait=tenacity.wait_fixed(1),
    )
async def regenerate_answer(question:str,documents:list[Document], history:ConversationHistory):
    return await recompose_chain.ainvoke({"documents":format_docs(documents), "question": question,"history":history})

async def get_hallucination_score(documents:list[Document], generated_answer:str):
    try:
        answer= await hallucination_chain.ainvoke({"documents": format_docs(documents), "generation": generated_answer})
        return answer
    except Exception:
        return "yes"

@tenacity.retry(
        stop=tenacity.stop_after_attempt(settings.MAX_LLM_RETRIES),
        wait=tenacity.wait_fixed(1),
    )
async def contextualize_query(question:str, history:ConversationHistory) -> str:
    try:
        query=await contexulizer_chain.ainvoke({"question":question, "history":conversation_history[-4:]})
    except Exception as e:
        query=question
    return query

async def answer_user(question:str) -> str:
    if conversation_history:
        query=await contextualize_query(question=question, history=conversation_history[-4:])
    else:
        query=question
    conversation_history.append(f"user question:{query}")
    documents = await retrieve(query)
    chosen_documents=await grade_documents(question=query, documents=documents)
    if len(chosen_documents)==0:
        generated_answer= "I apologize but I could not find an answer to your question."
    try:
        generated_answer = await generate_answer(question=query, documents=chosen_documents,history=conversation_history[-4:])
    except Exception:
        generated_answer= "I apologize but I could not find an answer to your question."
    hallucination_score = await get_hallucination_score(documents=chosen_documents,
                                                        generated_answer=generated_answer
                                                        )
    if "no" in hallucination_score:
        try:
            answer = await regenerate_answer(question=query, documents=chosen_documents,history=conversation_history[-4:])
        except Exception:
            return generated_answer
    else:
        answer = generated_answer
    conversation_history.append(f"AI answer:{answer}")
    return answer
