import chromadb
import logging
import colorlog
import os
import sys
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.evaluation import FaithfulnessEvaluator
from huggingface_hub import whoami
from dotenv import load_dotenv
from langchain_openai import OpenAI


load_dotenv()

log_colors = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(reset)s %(message)s",
    log_colors=log_colors,
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(handler)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logger.debug(f"running on {sys.platform}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load data from directory
input_dir = Path("C:/Users/lavml/Documents/Datasets/cooking/")
logger.info(f"huggingface user fullname: {whoami()['fullname']}")
logger.info(f"\n[loading data from {input_dir}]")

documents = SimpleDirectoryReader(input_dir=input_dir).load_data(show_progress=True)  # encoding="latin-1
logger.info(f"number_of_documents_created: {len(documents)}\n")
logger.debug(f"example:\n{documents[0]}")

# create vector store
logger.info(f"[creating croma vector store index]\n")
db = chromadb.PersistentClient(path="./itty_chroma_db_medium")
chroma_collection = db.get_or_create_collection("itty_cooking")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)

# Process documents one by one
logger.info("[Processing documents one by one]\n")
problematic_doc_id_ixs = []
for i, doc in enumerate(documents):
    doc_ix = i + 1
    doc_id_str = f"Document {doc_ix}"
    try:
        logger.debug(f"Processing {doc_id_str}: '{getattr(doc, 'metadata')['file_name']}'\n")

        # Step 1: Apply Sentence Splitting
        chunks = splitter.get_nodes_from_documents([doc])
        logger.debug(f"{doc_id_str}: {len(chunks)} chunks created")

        # Step 2: Embed each chunk
        for chunk in chunks:
            embedding = embed_model.get_text_embedding(chunk.text)
            chunk.embedding = embedding
        logger.debug(f"{doc_id_str}: Chunks successfully embedded")

        # Step 3: Store in Vector Store
        vector_store.add(chunks)
        logger.debug(f"{doc_id_str}: Successfully added to vector store\n")

    except UnicodeEncodeError as e:
        logger.error(f"UnicodeEncodeError in {doc_id_str}: {e}; skipping.")
        problematic_doc_id_ixs.append(doc_ix)

    except Exception as e:
        logger.error(f"Unexpected error in {doc_id_str}: {e}; skipping.")
        problematic_doc_id_ixs.append(doc_ix)

if problematic_doc_id_ixs:
    prob_file_names = [getattr(documents[prob_doc_ix], 'metadata')['file_name'] for prob_doc_ix in problematic_doc_id_ixs]
    logger.warning(
        f"problematic_documents:\n{problematic_doc_id_ixs}\n{prob_file_names}")
    # raise Exception("fix or remove problematic files manually to continue!")



# llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
llm = OpenAI()  # model="gpt-3.5-turbo"  # llama3:8b
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

# predict and evaluate
evaluator = FaithfulnessEvaluator()
query = "Mention some exotic main dish options from the eastern world that include potatoes as an ingredient."
response = query_engine.query(query)

logger.info(f"\n[augmented retrieval generation]")
logger.info(f"\nquery: \n{query}\n")
logger.info(f"\nresponse: \n{response}\n")

eval_result = evaluator.evaluate_response(response=response)

for k, v in eval_result.__dict__.items():
    if k=="response" or v is None:
        continue
    if k == "contexts":
        logger.info(f"{k}:")
        for i in v:
            logger.info(i)
    else:
        logger.info(f"{k}: {v}")
