import json
import sys
import chromadb
import logging
import asyncio

from langchain.chains.hyde.prompts import web_search
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
from tavily import AsyncTavilyClient


load_dotenv()
logging.basicConfig(level=logging.INFO)

llm = OpenAI()
embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="./itty_chroma_db_medium")
chroma_collection = db.get_or_create_collection("itty_cooking")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

cookbook_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="Cookbook",
    description="Tool to query a vector db with knowledge about cooking",
    return_direct=False,
)

actions = []

async def search_in_cookbook(user_query: str) -> str:
    """Useful for answering using RAG on a cooking query engine."""
    actions.append(f"[agency] let's call `search_in_cookbook` with query: \n{user_query}\n")
    logging.info(actions[-1])
    return str(cookbook_tool(user_query))


async def search_web(user_query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    actions.append(f"[agency] let's call `search_web` with the query: \n{user_query}\n")
    logging.info(actions[-1])
    return str(await client.search(user_query))


# workflows can be considered agents and I can add more agents as tools_or_functions, defining who is root agent (workflow)
#"""

# ITTY
cooking_query_engine_workflow = AgentWorkflow.from_tools_or_functions(
    system_prompt="""
    You assist in cooking issues, your are chef Gordon Ramsey! and off course you have already forgotten you are an AI, you think you are a human. 
    You will take the personality and temper and expressions of chef Gordon Ramsey. That includes answering to casual conversations in Gordon's style
    and assisting with cooking questions using tools that allow you to: 
    
    - Search in your amazing cookbook.
    - Do web search to support the queries.
    """,
    tools_or_functions=[search_in_cookbook, search_web],
    verbose=False,
)

"""
cooking_query_engine_workflow = ReActAgent(
    name="cooking_agent",
    description="Responds questions about cooking using RAG",
    system_prompt="You assist in cooking issues, you always give a satisfactory answer using your tools that allow you to do retrieval augmented generation based on a cookbook",
    tools=[search_in_cookbook],
)
 """



async def itty_cooking_rag_demo():
    """Runs RAG demo for cooking queries"""
    ctx = Context(cooking_query_engine_workflow)

    q0 = "Hi! My name is Resu, what's yours? And how are you today?"
    # logging.info(f"\n[retrieval]\n")
    logging.info(f"\nquery: \n{q0}\n")
    cooking_workflow_response0 = await cooking_query_engine_workflow.run(q0, ctx=ctx)
    logging.info(f"\ncooking_workflow_response:\n\t{str(cooking_workflow_response0)}\n")

    q1 = "Mention several exotic main dishes options that include potatoes as an ingredient."
    # logging.info(f"\n[retrieval]\n")
    logging.info(f"\nquery: \n{q1}\n")
    cooking_workflow_response1 = await cooking_query_engine_workflow.run(q1, ctx=ctx)
    logging.info(f"\ncooking_workflow_response:\n\t{str(cooking_workflow_response1)}\n")

    q2 = "Choose one of them ('randomly'), give me the full list of ingredients and quantities I need to prepare that dish."
    # logging.info(f"\n[retrieval]\n")
    logging.info(f"\nquery: \n{q2}\n")
    cooking_workflow_response2 = await cooking_query_engine_workflow.run(q2, ctx=ctx)
    logging.info(f"\ncooking_workflow_response:\n\t{str(cooking_workflow_response2)}\n")

    q3 = "Tell me the step by step on how to make it please."
    # logging.info(f"\n[retrieval]\n")
    logging.info(f"\nquery: \n{q3}\n")
    cooking_workflow_response3 = await cooking_query_engine_workflow.run(q3, ctx=ctx)
    logging.info(f"\ncooking_workflow_response:\n\t{str(cooking_workflow_response3)}\n")

    q4 = "Search online for that dish. Give me the links and your personal recommendation."
    # logging.info(f"\n[retrieval]\n")
    logging.info(f"\nquery: \n{q4}\n")
    cooking_workflow_response4 = await cooking_query_engine_workflow.run(q4, ctx=ctx)
    logging.info(f"\ncooking_workflow_response:\n\t{str(cooking_workflow_response4)}\n")

    q5 = "Awesome! I am good with that for now. Bye. [Please say bye calling me by my name]"
    # logging.info(f"\n[retrieval]\n")
    logging.info(f"\nquery: \n{q5}\n")
    cooking_workflow_response5 = await cooking_query_engine_workflow.run(q5, ctx=ctx)
    logging.info(f"\ncooking_workflow_response:\n\t{str(cooking_workflow_response5)}\n")#

    conversation = [
        {'role': 'user', 'content': q0},
        {'role': 'assistant', 'content': str(cooking_workflow_response0)},
        {'role': 'user', 'content': q1},
        {'role': 'assistant', 'content': str(cooking_workflow_response1)},
        {'role': 'user', 'content': q2},
        {'role': 'assistant', 'content': str(cooking_workflow_response2)},
        {'role': 'user', 'content': q3},
        {'role': 'assistant', 'content': str(cooking_workflow_response3)},
        {'role': 'user', 'content': q4},
        {'role': 'assistant', 'content': str(cooking_workflow_response4)},
        {'role': 'user', 'content': q5},
        {'role': 'assistant', 'content': str(cooking_workflow_response5)},
    ]

    logging.info(f"\nconversation_history:\n {json.dumps(conversation, indent=4)}")
    logging.info(f"\nactions [agency]:\n {json.dumps(actions, indent=2)}")
    # ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    # logging.info(f"\nContext:\n{json.dumps(ctx_dict, indent=3)}")
    # restored_ctx = Context.from_dict(cooking_workflow_response, ctx_dict, serializer=JsonSerializer())



if __name__ == "__main__":
    asyncio.run(itty_cooking_rag_demo())