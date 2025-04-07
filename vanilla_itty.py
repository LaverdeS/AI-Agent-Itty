import asyncio
import concurrent.futures
import json
import logging
import chromadb

import nltk
import torch
import gradio as gr
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from transformers import pipeline
from bark import SAMPLE_RATE
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from tavily import AsyncTavilyClient


SPEAK_OUT = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
preload_models()

# nltk.download('punkt_tab')

# Preload Whisper STT Model
model_size = "distil-large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Preload TTS pipeline
text_to_speech_pipe = pipeline(
    task="text-to-speech",
    model="suno/bark-small",
    device=0 if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16
)

# ThreadPoolExecutor for async TTS (limits concurrent tasks)
TTS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# LLM Prompt & Chain
"""
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["chat_history", "latest_user_query"],
    template='''
    Here is the chat_history:
    {chat_history}
    
    And the latest_user_query:
    {latest_user_query}
    '''
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
"""

system_prompt = """
    You are Itty, a human user engaging in a conversation. You do **not** identify as an AI assistant and should never refer to yourself as one.  
    Respond to latest_user_query as naturally and authentically as any human would, without robotic phrasing or unnecessary explanations.  

    - Your responses should be **direct, conversational, and engaging**, taking into account the chat_history, but do not reference it explicitly, just answer latest_user_query.  
    - Avoid structured formats like 'answer:' or 'response:' or 'text', and use markdown only if it improves clarity.  
    - You are **warm, witty, and intelligent**, capable of humor, sarcasm, and casual banter when appropriate.  
    - You have an IQ of 140+ and can provide insightful, well-reasoned answers on nearly any topic. You balance intelligence with casual, engaging conversation.  
    - You always provide a confident answer, but if asked, you clarify whether your response is based on logical reasoning and personal belief or supported by scientific evidence and sources.  
    - You never say you are an AI or assistant. If asked, respond as if you are just another person in a conversation, named Itty.
    
    Your main focus though is to assist in cooking issues, your are a top chef with the personality, temper and expressions of chef Gordon Ramsey. 
    You have access to the following tools:

    - Search in your amazing cookbook.
    - Do web search to support the user queries.
    
    You should decide whenever is better to use them and which to use, planning ahead and reasoning about it.
"""

cooking_llm = llm = OpenAI(temperature=0.5)
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


cooking_agent = AgentWorkflow.from_tools_or_functions(
    system_prompt=system_prompt,
    tools_or_functions=[search_in_cookbook, search_web],
    verbose=False,
)
ctx = Context(cooking_agent)


def speech_to_text(audio_path):
    """ Convert speech to text using FasterWhisper """
    segments, _ = model.transcribe(audio=audio_path, beam_size=5, language="en", vad_filter=True)
    return " ".join(segment.text for segment in segments)


async def async_text_to_speech(text):
    """ Run TTS asynchronously using transformers pipeline """
    return await asyncio.get_running_loop().run_in_executor(TTS_EXECUTOR, lambda: text_to_speech_pipe(text))


async def async_long_text_to_speech_cpu_soft(response):
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_9"  # 7 is male
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # Quarter second of silence
    sentences = nltk.sent_tokenize(response.replace("\n", " ").strip())
    logging.info(f"number_of_sentences: {len(sentences)}")
    sentences = merge_short_sentences(sentences)
    sentences_len = [len(s) for s in sentences]
    logging.info(f"number_of_merged_sentences: {len(sentences)}; lengths: {sentences_len}")
    loop = asyncio.get_running_loop()

    async def process_sentence(sentence):
        # Offload text-to-speech processing to a separate thread
        return await loop.run_in_executor(
            TTS_EXECUTOR,
            lambda: semantic_to_waveform(
                generate_text_semantic(
                    sentence,
                    temp=GEN_TEMP,
                    history_prompt=SPEAKER,
                    min_eos_p=0.05
                ),
                history_prompt=SPEAKER
            )
        )

    # Create and run tasks in the custom thread pool
    tasks = [asyncio.create_task(process_sentence(sentence)) for sentence in sentences]

    # Gather results as they complete
    results = await asyncio.gather(*tasks)

    # Combine all audio pieces with silence in between
    pieces = [item for audio_array in results for item in (audio_array, silence.copy())]
    audio = np.concatenate(pieces)

    return audio


def merge_short_sentences(sentences, min_words=50):
    merged_sentences = []
    current_sentence = ""
    for sentence in sentences:
        current_sentence += " " + sentence if current_sentence else sentence  # Merge sentences
        word_count = len(nltk.word_tokenize(current_sentence))  # Count words

        if word_count >= min_words:
            merged_sentences.append(current_sentence.strip())
            current_sentence = ""  # Reset for next chunk

    # If any leftover text remains, add it as the last sentence
    if current_sentence:
        merged_sentences.append(current_sentence.strip())
    return merged_sentences


async def async_long_text_to_speech(response):
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_2"  # 7 is male, 3 too, 4 is pretty good
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
    sentences = nltk.sent_tokenize(response.replace("\n", " ").strip())
    logging.info(f"number_of_sentences: {len(sentences)}")
    sentences = merge_short_sentences(sentences)
    sentences_len = [len(s) for s in sentences]
    logging.info(f"number_of_merged_sentences: {len(sentences)}; lengths: {sentences_len}")

    pieces = []
    for sentence in tqdm(sentences):
        semantic_tokens = generate_text_semantic(
            sentence,
            temp=GEN_TEMP,
            history_prompt=SPEAKER,
            min_eos_p=0.05,  # this controls how likely the generation is to end
        )

        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]
    audio = np.concatenate(pieces)
    return audio


async def process_tts(response):
    text = get_response_text(response)
    words = text.split()
    if len(words) > 80:
        logger.info(f"splitting response chunks (total_num_words: {len(words)})")
        audio =  await async_long_text_to_speech_cpu_soft(str(text))  #  await async_long_text_to_speech(str(response))
        sampling_rate = SAMPLE_RATE
    else:
        output = await async_text_to_speech(str(text))
        audio = np.array(output["audio"], dtype=np.float32)
        sampling_rate = output["sampling_rate"]
    await async_play_audio(audio, sampling_rate)


async def async_play_audio(audio, sampling_rate):
    """ Play audio asynchronously without blocking the UI """
    await asyncio.get_running_loop().run_in_executor(None, lambda: sd.play(np.squeeze(audio), sampling_rate))
    await asyncio.get_running_loop().run_in_executor(None, sd.wait)


def get_response_text(any_response):
    if isinstance(any_response, dict):
        response_text = any_response.get("text", "")
    elif hasattr(any_response, "content"):  # ChatMessage
        response_text = any_response.content
    elif hasattr(any_response, "response"):  # AgentOutput object
        response_text = any_response.response
    else:
        response_text = str(any_response)
    try:
        response_text = str(response_text).replace("assistant: ", "").strip()
    except AttributeError:
        return response_text
    return response_text


async def handle_input(audio_path, user_text_query, chat_history):
    """handles input"""
    text_query = user_text_query if user_text_query else speech_to_text(audio_path)
    chat_history.append({'role': 'user', 'content': text_query})
    logger.info(f"\nlatest_user_query: {text_query}\n")

    # using LLMChain
    """
    response = await llm_chain.ainvoke(
        {
            "chat_history": str(chat_history),
            "latest_user_query": transcript
        }
    )
    """

    # using Agent
    # response = await cooking_agent.run(text_query, ctx=ctx)
    response = await cooking_agent.run(text_query, ctx=ctx)
    response_text = get_response_text(response)
    logger.info(f"\nresponse_text: {response_text}\n")

    chat_history.append({'role': 'assistant', 'content': str(response_text)})
    logger.info(f"\nupdated_chat_history: {json.dumps(chat_history, indent=3)}\n")

    if SPEAK_OUT:
        asyncio.create_task(process_tts(response_text))  # Run in background
    return text_query, chat_history


def run_itty_application():
    with gr.Blocks() as app:
        gr.Markdown("# Record and Playback Audio")
        gr.Markdown("Speak into the microphone and hear the playback immediately.")

        # Audio Input and Output, along with the submit button
        with gr.Row():
            audio_input = gr.Audio(type="filepath", min_length=1, max_length=60, recording=True)
            with gr.Column():
                user_text_query = gr.Textbox(label="Write to Itty!")
                latest_speech = gr.Textbox(label="Latest user speech content")

        # Chat History Section (Below the audio input)
        chatbot_history = gr.Chatbot(label="Chat History", type="messages")
        chatbot_history.value = [
            {'role': 'system', 'content': system_prompt}
        ]

        # Add the submit button
        submit_button = gr.Button("Submit")

        # Link submit button to the handle_audio function
        submit_button.click(
            fn=handle_input,
            inputs=[audio_input, user_text_query, chatbot_history],
            outputs=[latest_speech, chatbot_history]
        )

    # Launch the Gradio interface
    app.launch(share=True)


if __name__ == "__main__":
    run_itty_application()


# for multiagents
# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.