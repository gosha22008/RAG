"""FastAPI с RAG-пайплайном.
Запуск: uvicorn api.server:app --host 0.0.0.0 --port 10000
"""
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from llama_index.core import Settings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

# Подключаем код проекта
sys.path.append(str(Path(__file__).parent.parent))
from experiments.rag_exp_new import (
    RAGConfig, ChunkingConfig, EmbeddingConfig, RetrievalConfig, RerankConfig,
    LLMConfig, _index_documents, build_retriever, build_reranker, LLMGenerator,
    load_and_prepare_docs, PageMapper, EvalDataLoader, SplitterFactory
)
# Подгружай свои функции для page_mapper и splitter_factory
# from experiments.utils import page_mapper, splitter_factory  # как у тебя называется
docs = load_and_prepare_docs("./data")
page_mapper = PageMapper(docs)
# eval_data = EvalDataLoader.load("./context/docs_questions_qwen3_14b_awq.jsonl",
#                                 skip_pages={72, 73, 74, 75, 76, 77, 78, 79, 80, 81})
splitter_factory = SplitterFactory("BAAI/bge-m3")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Конфигурация ===
Settings.llm = None
Settings.embed_model = OpenAILikeEmbedding(
    model_name="qwen3-embed",
    api_base="http://localhost:8081/v1",
)

cfg_rag = RAGConfig(
    name="best_pipeline",
    retrieval=RetrievalConfig(mode="hybrid", top_k_values=[10]),
    chunking=ChunkingConfig(splitter_type="hierarchical", hierarchy_sizes=[512, 256, 128]),
    embedding=EmbeddingConfig(model_name="BAAI/bge-m3", api_model_name="qwen3-embed"),
    rerank=RerankConfig(
        enabled=True,
        model_name="BAAI/bge-reranker-v2-m3",
        backend="tei",
        top_n=5,
    ),
    recreate=False,
)

llm_cfg = LLMConfig(
    api_base="http://localhost:8000/v1",
    api_model_name="qwen3-4b",
    max_tokens=1024,
)

# === Глобальные объекты пайплайна (ленивая инициализация) ===
class Pipeline:
    def __init__(self):
        self.retriever = None
        self.reranker = None
        self.llm = None

    def init(self):
        logger.info("Инициализация RAG пайплайна...")
        client = QdrantClient(prefer_grpc=True)
        # ВАЖНО: подставь свои page_mapper и splitter_factory
        index, nodes = _index_documents(cfg_rag, page_mapper, splitter_factory, client)
        self.retriever = build_retriever(index, cfg_rag.retrieval, nodes=nodes)
        self.reranker = build_reranker(cfg_rag.rerank)
        self.llm = LLMGenerator(llm_cfg)
        logger.info("Пайплайн готов")

    def answer(self, query: str) -> dict:
        """Полный pipeline: retrieve → rerank → generate."""
        candidates = self.retriever.retrieve(query)
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)
        contexts = [n.get_content() for n in candidates[:5]]
        pages = [n.metadata.get("pages_covered", []) for n in candidates[:5]]
        answer = self.llm.generate(query, contexts)
        return {
            "answer": answer,
            "pages": pages,
            "num_contexts": len(contexts),
        }


pipeline = Pipeline()


# === FastAPI ===
app = FastAPI(title="RAG API")


@app.on_event("startup")
def startup():
    pipeline.init()


class Query(BaseModel):
    text: str


class Answer(BaseModel):
    answer: str
    pages: list
    num_contexts: int


@app.get("/")
def root():
    return {"status": "ok", "service": "RAG API"}


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    """Endpoint для задавания вопроса по корпусу."""
    logger.info(f"Запрос: {query.text[:80]}")
    try:
        result = pipeline.answer(query.text)
        return Answer(**result)
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return Answer(answer=f"Ошибка: {e}", pages=[], num_contexts=0)