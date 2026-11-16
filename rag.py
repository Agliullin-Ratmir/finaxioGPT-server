import hashlib
import json
import os
import traceback
import asyncio
from pathlib import Path
import uvicorn
import chromadb
from openai import OpenAI, InternalServerError, UnprocessableEntityError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# LlamaIndex core & nodes
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, PromptTemplate
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# LlamaIndex integrations
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import MarkdownNodeParser

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT_DIR / "docs")).resolve()
SKILLS_FILE = Path(os.getenv("RAG_SKILLS_FILE", ROOT_DIR / "skills.md")).resolve()
DB_PATH = Path(os.getenv("RAG_DB_PATH", ROOT_DIR / "chroma_db")).resolve()
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "reactivegpt_docs")
MODEL_NAME = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
MAX_TOKENS = int(os.getenv("NVIDIA_MAX_TOKENS", "8192"))
HOST = os.getenv("RAG_HOST", "0.0.0.0")
PORT = int(os.getenv("RAG_PORT", "3050"))
LLM_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_CONFIG_FILE = Path(os.getenv("RAG_API_CONFIG", ROOT_DIR / "api.md")).resolve()
EMBEDDING_MODEL_NAME = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-8b")
INDEX_FORMAT_VERSION = 1

# ---------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------
query_engine = None


class NVIDIAEmbedding(BaseEmbedding):
    """Use NVIDIA's OpenAI-compatible embeddings endpoint with arbitrary model IDs."""

    api_client: object = Field(exclude=True)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            result = self.api_client.embeddings.create(
                model=self.model_name,
                input=texts,
                extra_body={"input_type": input_type},
            )
            return [item.embedding for item in sorted(result.data, key=lambda item: item.index)]
        except InternalServerError as error:
            if error.status_code != 504:
                raise
            if len(texts) > 1:
                midpoint = len(texts) // 2
                print(
                    f"⚠️ NVIDIA embedding request timed out for {len(texts)} texts; retrying in smaller batches.",
                    flush=True,
                )
                return (
                    self._embed(texts[:midpoint], input_type)
                    + self._embed(texts[midpoint:], input_type)
                )
            print(
                "⚠️ NVIDIA timed out embedding one text; using a neutral embedding for it.",
                flush=True,
            )
            return [[0.0] * 4096]
        except UnprocessableEntityError as error:
            if "content safety" not in str(error).lower():
                raise
            if len(texts) > 1:
                midpoint = len(texts) // 2
                return (
                    self._embed(texts[:midpoint], input_type)
                    + self._embed(texts[midpoint:], input_type)
                )
            print(
                "⚠️ NVIDIA safety screening rejected one text chunk; using a neutral embedding for it.",
                flush=True,
            )
            return [[0.0] * 4096]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query], input_type="query")[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text], input_type="passage")[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), self.embed_batch_size):
            embeddings.extend(self._embed(texts[start:start + self.embed_batch_size], input_type="passage"))
        return embeddings

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)


def load_api_config() -> dict:
    config = {}
    if API_CONFIG_FILE.is_file():
        for line in API_CONFIG_FILE.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                config[key.strip()] = value.strip().strip("\"'")
    return config


def get_embedding_model_name() -> str:
    model = os.getenv("NVIDIA_EMBEDDING_MODEL") or load_api_config().get("model") or EMBEDDING_MODEL_NAME
    return model if "/" in model else f"nvidia/{model}"


def documents_fingerprint() -> str:
    """Hash supported source files so unchanged documentation can reuse Chroma."""
    digest = hashlib.sha256()
    supported_extensions = {".md", ".sol", ".pdf"}
    paths = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    for path in paths:
        digest.update(path.relative_to(DATA_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    embedding_model = get_embedding_model_name()
    digest.update(f"{embedding_model}:{INDEX_FORMAT_VERSION}".encode("utf-8"))
    return digest.hexdigest()


def load_cached_node_dict(cache_path: Path):
    serialized_nodes = json.loads(cache_path.read_text(encoding="utf-8"))
    nodes = [TextNode.model_validate(node) for node in serialized_nodes]
    if not nodes:
        raise ValueError("Saved node cache is empty")
    return {node.node_id: node for node in nodes}


def save_cached_node_dict(cache_path: Path, node_dict):
    serialized_nodes = [node.model_dump(mode="json") for node in node_dict.values()]
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(serialized_nodes), encoding="utf-8")
    temporary_path.replace(cache_path)

# ---------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="The developer's question or request")
    top_k: Optional[int] = Field(default=5, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(default=0.1, description="Model temperature for generation")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The generated answer from the LLM")

app = FastAPI(title="Reactive Network Advanced RAG", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_llamaindex():
    global query_engine

    api_config = load_api_config()
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY") or api_config.get("apiKey")
    embedding_model = get_embedding_model_name()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {DATA_DIR}")
    if not SKILLS_FILE.is_file():
        raise FileNotFoundError(f"Skills file does not exist: {SKILLS_FILE}")

    # Use the chat model for answers and NVIDIA's configured embedding model for retrieval.
    Settings.llm = OpenAILike(
        model=MODEL_NAME,
        api_key=api_key or "not-configured",
        api_base=LLM_BASE_URL,
        is_chat_model=True,
        context_window=32768,
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        timeout=300.0
    )
    Settings.embed_model = NVIDIAEmbedding(
        model_name=embedding_model,
        api_client=OpenAI(
            api_key=api_key or "not-configured",
            base_url=LLM_BASE_URL,
            timeout=300.0,
        ),
        embed_batch_size=8,
    )

    DB_PATH.mkdir(parents=True, exist_ok=True)
    fingerprint = documents_fingerprint()
    node_cache_path = DB_PATH / "rag_nodes.json"
    manifest_path = DB_PATH / "rag_manifest.json"
    db = chromadb.PersistentClient(path=str(DB_PATH))
    collections = db.list_collections()
    collection_exists = any(
        getattr(collection, "name", collection) == COLLECTION_NAME
        for collection in collections
    )

    # Reuse the persisted Chroma vectors and parent-node lookup when source files
    # and indexing configuration have not changed.
    all_nodes_dict = None
    index = None
    if collection_exists:
        chroma_collection = db.get_collection(COLLECTION_NAME)
        collection_count = chroma_collection.count()
        if collection_count > 0 and manifest_path.is_file() and node_cache_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("fingerprint") == fingerprint
                    and manifest.get("collection_name") == COLLECTION_NAME
                ):
                    all_nodes_dict = load_cached_node_dict(node_cache_path)
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    index = VectorStoreIndex.from_vector_store(vector_store)
                    print(
                        f"♻️ Reusing saved vector index ({chroma_collection.count()} vectors).",
                        flush=True,
                    )
            except Exception as error:
                print(f"⚠️ Saved index cache could not be reused: {error}", flush=True)
        elif collection_count > 0 and not manifest_path.exists() and not node_cache_path.exists():
            # Migrate an index created by the previous version without re-embedding it.
            try:
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                stored_nodes = vector_store.get_nodes(node_ids=None)
                all_nodes_dict = {
                    node.node_id: node
                    for node in stored_nodes
                    if isinstance(node, TextNode) and not isinstance(node, IndexNode)
                }
                missing_parent_ids = {
                    node.index_id
                    for node in stored_nodes
                    if isinstance(node, IndexNode) and node.index_id not in all_nodes_dict
                }
                if not all_nodes_dict or missing_parent_ids:
                    raise ValueError(
                        f"Existing index is missing {len(missing_parent_ids)} parent nodes"
                    )
                index = VectorStoreIndex.from_vector_store(vector_store)
                save_cached_node_dict(node_cache_path, all_nodes_dict)
                manifest = {
                    "fingerprint": fingerprint,
                    "collection_name": COLLECTION_NAME,
                }
                temporary_manifest = manifest_path.with_suffix(".json.tmp")
                temporary_manifest.write_text(json.dumps(manifest), encoding="utf-8")
                temporary_manifest.replace(manifest_path)
                print(
                    f"♻️ Migrated the existing vector index ({collection_count} vectors) without re-embedding.",
                    flush=True,
                )
            except Exception as error:
                print(f"⚠️ Existing vector index could not be migrated: {error}", flush=True)
                all_nodes_dict = None
                index = None

    if index is None:
        if collection_exists:
            db.delete_collection(COLLECTION_NAME)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print(f"📚 Reading documents from {DATA_DIR}...", flush=True)
        reader = SimpleDirectoryReader(
            input_dir=str(DATA_DIR), required_exts=[".md", ".sol", ".pdf"], recursive=True
        )
        documents = reader.load_data()
        if not documents:
            raise RuntimeError(f"No supported documents found under {DATA_DIR}")
        print(f"📄 Loaded {len(documents)} documents; splitting into nodes...", flush=True)

        # Parse and chunk locally; document contents are not sent to NVIDIA during indexing.
        md_parser = MarkdownNodeParser()
        sol_parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=150, separator="\n}\n")
        child_parser = TokenTextSplitter(chunk_size=256, chunk_overlap=50)
        all_nodes = []

        for doc in documents:
            suffix = Path(doc.metadata.get("file_path", "")).suffix.lower()
            if suffix == ".md":
                all_nodes.extend(md_parser.get_nodes_from_documents([doc]))
            else:
                parent_nodes = sol_parser.get_nodes_from_documents([doc])
                for parent_node in parent_nodes:
                    child_nodes = child_parser.get_nodes_from_documents([parent_node])
                    for child in child_nodes:
                        all_nodes.append(IndexNode.from_text_node(child, parent_node.node_id))
                    all_nodes.append(parent_node)

        if not all_nodes:
            raise RuntimeError(f"No indexable text found under {DATA_DIR}")
        print(
            f"🧩 Created {len(all_nodes)} nodes; building vector index (embedding progress follows)...",
            flush=True,
        )

        all_nodes_dict = {
            node.node_id: node
            for node in all_nodes
            if isinstance(node, TextNode) and not isinstance(node, IndexNode)
        }
        index = VectorStoreIndex(all_nodes, storage_context=storage_context, show_progress=True)
        save_cached_node_dict(node_cache_path, all_nodes_dict)
        manifest = {
            "fingerprint": fingerprint,
            "collection_name": COLLECTION_NAME,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        temporary_manifest.replace(manifest_path)
        print("✅ Vector index built and saved.", flush=True)

    # 6. Ретривер (с поддержкой табличных узлов)
    vector_retriever = index.as_retriever(similarity_top_k=6)
    retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=all_nodes_dict,
        verbose=True
    )

    # 7. Финал
    with open(SKILLS_FILE, 'r', encoding='utf-8') as f:
        skills = f.read().strip()

    custom_prompt = PromptTemplate(
        f"<system_instructions>\n{skills}\n</system_instructions>\n"
        "<context>\n{context_str}\n</context>\n"
        "<user_query>\n{query_str}\n</user_query>\n"
        "<response_guidelines>\n"
        "1. Веди диалог на русском языке: формулируй и интерпретируй запросы на русском, ответы всегда давай на русском.\n"
        "2. При анализе таблиц сопоставляй заголовки со значениями.\n"
        "3. Provide full, compilable Solidity code.\n"
        "</response_guidelines>"
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        text_qa_template=custom_prompt
    )

@app.on_event("startup")
async def startup_event():
    print(f"🚀 Initializing Advanced RAG with {MODEL_NAME}...")
    try:
        setup_llamaindex()
        print("✅ Recursive Retrieval System Ready.")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()

@app.post("/send-request", response_model=QueryResponse)
def send_query(request: QueryRequest):
    try:
        if query_engine is None:
            raise HTTPException(status_code=503, detail="System not initialized")
        if not (os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY") or load_api_config().get("apiKey")):
            raise HTTPException(status_code=503, detail="Set NVIDIA_API_KEY before sending queries")

        response = query_engine.query(request.query)
        return JSONResponse(content={"response": str(response)})

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
    
    # https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
