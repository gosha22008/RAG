"""
RAG Experiment Framework (v3)
==============================
Простой синхронный фреймворк для экспериментов с RAG.

Поддерживает:
- Чанкинг: token / sentence / semantic / hierarchical
- Retrieval: dense / sparse (BM25) / hybrid (fusion)
- Reranker через HTTP API (vLLM или TEI)
- Matryoshka embeddings (через truncate_dim)
- gRPC для Qdrant

Пример использования:

    from rag_exp import (
        RAGConfig, RetrievalConfig, RerankConfig,
        load_and_prepare_docs, PageMapper, EvalDataLoader,
        SplitterFactory, run_experiments, ResultsVisualizer,
    )

    docs = load_and_prepare_docs("../data")
    page_mapper = PageMapper(docs)
    eval_data = EvalDataLoader.load("../context/questions.jsonl")
    splitter_factory = SplitterFactory("Qwen/Qwen3-Embedding-0.6B")

    configs = [
        RAGConfig(name="dense_baseline"),
        RAGConfig(
            name="dense_rerank",
            rerank=RerankConfig(enabled=True),
        ),
        RAGConfig(
            name="hybrid_rerank",
            retrieval=RetrievalConfig(mode="hybrid", retrieval_k=20),
            rerank=RerankConfig(enabled=True),
        ),
    ]

    df = run_experiments(configs, page_mapper, splitter_factory, eval_data)
    ResultsVisualizer.compare_categories(df, category_col="experiment")
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from llama_index.core import (
    VectorStoreIndex, Document, Settings,
    StorageContext, SimpleDirectoryReader,
)
from llama_index.core.node_parser import (
    SentenceSplitter, TokenTextSplitter,
    SemanticSplitterNodeParser, HierarchicalNodeParser,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("rag")


# ═══════════════════════════════════════════════════════════════════
# 1. КОНФИГИ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChunkingConfig:
    """Параметры разбиения документов на чанки."""
    chunk_size: int = 500
    chunk_overlap: int = 200
    splitter_type: str = "token"  # token | sentence | semantic | hierarchical
    breakpoint_percentile: int = 90
    hierarchy_sizes: List[int] = field(default_factory=lambda: [1024, 512, 128])


@dataclass
class EmbeddingConfig:
    """Параметры embedding-модели через OpenAI-like API (vLLM/TEI)."""
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    api_base: str = "http://localhost:8081/v1"
    api_model_name: str = "qwen3-embed"
    params: Optional[dict] = None
    truncate_dim: Optional[int] = None  # Matryoshka: обрезать до N размерностей


# @dataclass
# class RerankConfig:
#     """Параметры реранкера через HTTP API.

#     backend = "vllm" — формат Cohere-style (POST {api_base}/v1/rerank)
#     backend = "tei"  — формат TEI         (POST {api_base}/rerank)
#     """
#     enabled: bool = False
#     model_name: str = "Qwen/Qwen3-Reranker-0.6B"
#     api_base: str = "http://localhost:8082"
#     api_model_name: str = "qwen3-rerank"
#     top_n: int = 5
#     backend: str = "vllm"  # vllm | tei

@dataclass
class RerankConfig:
    enabled: bool = False
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    top_n: int = 5
    device: str = "cuda"
    backend: str = "local"
    prompt_template: str = None  # None для стандартных, "qwen3" для Qwen3-Reranker
    api_base: str = "http://localhost:8082"
    api_model_name: str = "harr-rerank"


@dataclass
class RetrievalConfig:
    """Параметры поиска и оценки."""
    mode: str = "dense"  # dense | sparse | hybrid
    retrieval_k: int = 15  # сколько кандидатов достать ДО реранкинга
    top_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])


@dataclass
class RAGConfig:
    """Полная конфигурация одного эксперимента."""
    name: str = "baseline"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    qdrant_url: str = "http://127.0.0.1:6333"
    recreate: bool = True

    @property
    def collection_name(self) -> str:
        return f"wb_{self.name}"


# ═══════════════════════════════════════════════════════════════════
# 2. ЗАГРУЗКА ДОКУМЕНТОВ И ВОПРОСОВ
# ═══════════════════════════════════════════════════════════════════

def load_and_prepare_docs(data_dir: str) -> List[Document]:
    """Загружает PDF из директории, нумерует страницы, удаляет пустые."""
    reader = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".pdf"])
    docs = reader.load_data()
    for i, doc in enumerate(docs):
        doc.metadata["page_label"] = i + 1

    empty = [i for i, d in enumerate(docs) if not d.text]
    if empty:
        logger.info(f"Пустые страницы: {[docs[i].metadata['page_label'] for i in empty]}")
        for i in sorted(empty, reverse=True):
            docs.pop(i)

    logger.info(f"Загружено {len(docs)} непустых страниц")
    return docs


class PageMapper:
    """Склеивает все документы в один длинный текст и хранит карту страниц.

    Зачем: после чанкинга нам нужно знать, на каких страницах лежит каждый чанк
    (для подсчёта метрик, где истина — номера страниц).
    """

    def __init__(self, docs: List[Document]):
        self.full_text, self.page_ranges = self._build(docs)
        # Один большой документ для подачи в любой splitter
        self.document = Document(text=self.full_text)

    @staticmethod
    def _build(docs: List[Document]):
        text, ranges = "", []
        for doc in docs:
            start = len(text)
            text += doc.text + "\n\n"
            ranges.append((start, len(text), str(doc.metadata["page_label"])))
        return text, ranges

    def get_pages(self, start: int, end: int) -> List[str]:
        """Возвращает страницы, пересекающиеся с диапазоном [start, end)."""
        return [page for s, e, page in self.page_ranges if start < e and end > s]

    def enrich_nodes(self, nodes):
        """Добавляет в metadata каждого node поле pages_covered."""
        pos = 0
        for node in nodes:
            idx = self.full_text.find(node.text, pos)
            if idx == -1:
                idx = self.full_text.find(node.text)
            if idx != -1:
                node.metadata["pages_covered"] = self.get_pages(idx, idx + len(node.text))
                pos = idx
            else:
                node.metadata["pages_covered"] = []
        return nodes


class EvalDataLoader:
    """Загрузка вопросов из jsonl-файла."""

    @staticmethod
    def load(path: str,
             question_types: Optional[List[str]] = None,
             skip_pages: Optional[Set[int]] = None) -> List[Dict]:
        question_types = question_types or ["simple", "medium", "complex"]
        skip_pages = skip_pages or set()

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                key = list(entry.keys())[0]
                questions = list(entry.values())[0]
                pages = key.split("_")[1:]

                if any(int(p) in skip_pages for p in pages):
                    continue

                for qt in question_types:
                    if qt in questions and questions[qt]:
                        data.append({
                            "query": questions[qt],
                            "relevant_pages": pages,
                            "question_type": qt,
                        })

        logger.info(f"Загружено {len(data)} вопросов из {path}")
        return data


# ═══════════════════════════════════════════════════════════════════
# 3. EMBEDDING (через OpenAI-like API)
# ═══════════════════════════════════════════════════════════════════

def setup_embedding(cfg: EmbeddingConfig) -> None:
    """Устанавливает Settings.embed_model. Безопасно при повторных вызовах:
    если модель уже та, что нужно — ничего не делает.
    """
    cur = Settings._embed_model
    if cur is not None and getattr(cur, "model_name", None) == cfg.api_model_name:
        return

    # Если модель поддерживает Matryoshka — передаём dimensions через additional_kwargs
    extra = dict(cfg.params or {})
    if cfg.truncate_dim is not None:
        extra["dimensions"] = cfg.truncate_dim

    Settings.embed_model = OpenAILikeEmbedding(
        model_name=cfg.api_model_name,
        api_base=cfg.api_base,
        additional_kwargs=extra,
    )
    suffix = f" (dim={cfg.truncate_dim})" if cfg.truncate_dim else ""
    logger.info(f"Embedding: {cfg.model_name} @ {cfg.api_base}{suffix}")


# ═══════════════════════════════════════════════════════════════════
# 4. ЧАНКИНГ
# ═══════════════════════════════════════════════════════════════════

class SplitterFactory:
    """Создаёт сплиттеры по конфигу. Тип задаётся через ChunkingConfig.splitter_type."""

    def __init__(self, embed_model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        logger.info(f"Токенизатор: {embed_model_name}")

    def create(self, cfg: ChunkingConfig):
        t = cfg.splitter_type

        if t == "token":
            return TokenTextSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                tokenizer=self._tokenizer.encode,
            )
        if t == "sentence":
            return SentenceSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                tokenizer=self._tokenizer.encode,
            )
        if t == "semantic":
            assert Settings.embed_model is not None, \
                "Перед созданием semantic splitter вызови setup_embedding()"
            return SemanticSplitterNodeParser(
                embed_model=Settings.embed_model,
                breakpoint_percentile_threshold=cfg.breakpoint_percentile,
            )
        if t == "hierarchical":
            return HierarchicalNodeParser.from_defaults(chunk_sizes=cfg.hierarchy_sizes)

        raise ValueError(f"Неизвестный splitter_type: {t}")


# ═══════════════════════════════════════════════════════════════════
# 5. ИНДЕКСАЦИЯ
# ═══════════════════════════════════════════════════════════════════

def build_index(client: QdrantClient,
                collection_name: str,
                nodes: Optional[list] = None,
                recreate: bool = True) -> VectorStoreIndex:
    """Создаёт или открывает Qdrant-коллекцию.

    - Если nodes переданы — индексирует их (создаёт коллекцию).
    - Если nodes=None — открывает существующую коллекцию для поиска.
    """
    if nodes is not None:
        if recreate and client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        if not client.collection_exists(collection_name):
            vs = QdrantVectorStore(collection_name=collection_name, client=client)
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=StorageContext.from_defaults(vector_store=vs),
                show_progress=True,
            )
            logger.info(f"{collection_name}: проиндексировано {len(nodes)} чанков")
            return index

    # Открываем существующую коллекцию
    vs = QdrantVectorStore(collection_name=collection_name, client=client)
    return VectorStoreIndex.from_vector_store(vector_store=vs)


# ═══════════════════════════════════════════════════════════════════
# 6. РЕТРИВЕРЫ (dense / sparse / hybrid)
# ═══════════════════════════════════════════════════════════════════

def build_retriever(index: VectorStoreIndex,
                    retrieval_cfg: RetrievalConfig,
                    nodes: Optional[list] = None):
    """Возвращает retriever нужного типа.

    - dense  — векторный поиск через Qdrant (nodes не нужны)
    - sparse — BM25 (nodes обязательны)
    - hybrid — fusion dense + BM25 через reciprocal rank (nodes обязательны)
    """
    mode = retrieval_cfg.mode
    k = retrieval_cfg.retrieval_k

    if mode == "dense":
        return index.as_retriever(similarity_top_k=k)

    if mode == "sparse":
        assert nodes is not None, "Для sparse retrieval нужны nodes"
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k)

    if mode == "hybrid":
        assert nodes is not None, "Для hybrid retrieval нужны nodes"
        dense = index.as_retriever(similarity_top_k=k)
        sparse = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k)
        return QueryFusionRetriever(
            retrievers=[dense, sparse],
            similarity_top_k=k,
            num_queries=1,           # без расширения запроса
            mode="reciprocal_rerank",
            use_async=False,
        )

    raise ValueError(f"Неизвестный retrieval mode: {mode}")


# ═══════════════════════════════════════════════════════════════════
# 7. РЕРАНКЕР (локальный через CrossEncoder ИЛИ через HTTP API)
# ═══════════════════════════════════════════════════════════════════
class JinaV3Reranker:
    """Реранкер через jinaai/jina-reranker-v3 (listwise, локально).
    
    Особенность: принимает query + все документы одним вызовом,
    использует model.rerank(), а не CrossEncoder.predict().
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v3",
                 top_n: int = 5, device: str = "cuda"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            model_name, dtype="auto", trust_remote_code=True,
        ).to(device).eval()
        self.top_n = top_n
        self.device = device
        logger.info(f"Jina V3 Reranker loaded: {model_name} on {device}")

    def rerank(self, query: str, nodes: list) -> list:
        if not nodes:
            return nodes

        texts = [n.get_content() for n in nodes]
        results = self.model.rerank(query, texts, top_n=self.top_n)

        reranked = []
        for r in results:
            node = nodes[r["index"]]
            node.score = float(r["relevance_score"])
            reranked.append(node)
        return reranked

    def unload(self):
        del self.model
        import gc, torch
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Jina V3 Reranker unloaded")


class LLMReranker:
    """Реранкер на основе generative LLM (Qwen3-Reranker, FRIDA и т.д.).
    
    Модель получает prompt с query+doc и отвечает yes/no.
    Score = вероятность токена 'yes'.
    """

    def __init__(self, model_name: str, top_n: int = 5, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.top_n = top_n
        self.device = device
        self._yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        logger.info(f"LLM Reranker loaded: {model_name} on {device}")

    def _score_pair(self, query: str, document: str) -> float:
        """Считает score для одной пары (query, document)."""
        import torch

        prompt = (
            f"Is the following document relevant to the query?\n"
            f"Query: {query}\n"
            f"Document: {document}\n"
            f"Answer yes or no:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Берём логиты последнего токена
            logits = outputs.logits[0, -1, :]
            # Вероятность "yes" через softmax
            probs = torch.softmax(logits, dim=-1)
            score = probs[self._yes_token_id].item()
        return score

    def rerank(self, query: str, nodes: list) -> list:
        if not nodes:
            return nodes

        scored = []
        for node in nodes:
            score = self._score_pair(query, node.get_content())
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for node, score in scored[:self.top_n]:
            node.score = float(score)
            reranked.append(node)
        return reranked

    def unload(self):
        del self.model, self.tokenizer
        import gc, torch
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("LLM Reranker unloaded")


class LocalReranker:
    """Реранкер через sentence-transformers (локально, CPU или GPU).
    
    Поддерживает:
    - Стандартные cross-encoders (BGE, Jina v2) — пары (query, doc) как есть
    - Qwen3-Reranker — с специальным промпт-шаблоном
    """

    def __init__(self, model_name: str, top_n: int = 5, 
                 device: str = "cuda", prompt_template: str = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, trust_remote_code=True)
        self.top_n = top_n
        self.device = device
        self.prompt_template = prompt_template
        logger.info(f"Reranker loaded: {model_name} on {device}")

    def _format_pair(self, query: str, document: str) -> list:
        """Форматирует пару (query, doc) с учётом шаблона модели."""
        if self.prompt_template == "qwen3":
            prefix = (
                '<|im_start|>system\n'
                'Judge whether the Document meets the requirements based on '
                'the Query and the Instruct provided. Note that the answer '
                'can only be "yes" or "no".<|im_end|>\n'
                '<|im_start|>user\n'
                '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n'
            )
            suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            formatted_query = f"{prefix}<Query>: {query}\n"
            formatted_doc = f"<Document>: {document}{suffix}"
            return [formatted_query, formatted_doc]
        
        # Стандартные cross-encoders — пара как есть
        return [query, document]

    def rerank(self, query: str, nodes: list) -> list:
        if not nodes:
            return nodes

        pairs = [self._format_pair(query, n.get_content()) for n in nodes]
        scores = self.model.predict(pairs)

        scored = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)

        reranked = []
        for node, score in scored[:self.top_n]:
            node.score = float(score)
            reranked.append(node)
        return reranked

    def unload(self):
        del self.model
        import gc, torch
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Reranker unloaded")


class HTTPReranker:
    """Реранкер через HTTP API (vLLM или TEI).

    backend:
      - "vllm_score": vLLM с --task score (POST /v1/score)
      - "vllm_rerank": vLLM Cohere-style (POST /v1/rerank)
      - "tei": TEI (POST /rerank)
    """

    def __init__(self, api_base: str, model_name: str,
                 top_n: int = 5, backend: str = "vllm_score"):
        assert backend in ("vllm_score", "vllm_rerank", "tei")
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.top_n = top_n
        self.backend = backend

    def rerank(self, query: str, nodes: list) -> list:
        if not nodes:
            return nodes

        texts = [n.get_content() for n in nodes]

        if self.backend == "vllm_score":
            url = f"{self.api_base}/v1/score"
            payload = {
                "model": self.model_name,
                "text_1": query,
                "text_2": texts,
            }
            data = requests.post(url, json=payload, timeout=60).json()
            results = sorted(data["data"], key=lambda x: x["score"], reverse=True)
            score_key = "score"

        elif self.backend == "vllm_rerank":
            url = f"{self.api_base}/v1/rerank"
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": texts,
                "top_n": len(texts),
            }
            data = requests.post(url, json=payload, timeout=60).json()
            results = data.get("results", [])
            score_key = "relevance_score"

        elif self.backend == "tei":
            url = f"{self.api_base}/rerank"
            payload = {"query": query, "texts": texts}
            results = requests.post(url, json=payload, timeout=60).json()
            score_key = "score"

        reranked = []
        for item in results[:self.top_n]:
            node = nodes[item["index"]]
            node.score = float(item[score_key])
            reranked.append(node)
        return reranked

    def unload(self):
        """Для совместимости интерфейса с LocalReranker. Ничего не делает."""
        pass


def build_reranker(cfg: RerankConfig):
    if not cfg.enabled:
        return None

    if cfg.backend == "local":
        return LocalReranker(
            model_name=cfg.model_name,
            top_n=cfg.top_n,
            device=cfg.device,
            prompt_template=cfg.prompt_template,
        )
    elif cfg.backend == "local_llm":
        return LLMReranker(
            model_name=cfg.model_name,
            top_n=cfg.top_n,
            device=cfg.device,
        )
    elif cfg.backend == "jina_v3":
        return JinaV3Reranker(
            model_name=cfg.model_name,
            top_n=cfg.top_n,
            device=cfg.device,
        )
    else:
        return HTTPReranker(
            api_base=cfg.api_base,
            model_name=cfg.api_model_name,
            top_n=cfg.top_n,
            backend=cfg.backend,
        )


# def build_reranker(cfg: RerankConfig):
#     """Создаёт реранкер по конфигу. Возвращает None если реранкинг выключен."""
#     if not cfg.enabled:
#         return None

#     if cfg.backend == "local":
#         return LocalReranker(
#             model_name=cfg.model_name,
#             top_n=cfg.top_n,
#             device=cfg.device,
#         )
#     else:
#         return HTTPReranker(
#             api_base=cfg.api_base,
#             model_name=cfg.api_model_name,
#             top_n=cfg.top_n,
#             backend=cfg.backend,
#         )


# # ═══════════════════════════════════════════════════════════════════
# # 7. РЕРАНКЕР (через HTTP API: vLLM или TEI)
# # ═══════════════════════════════════════════════════════════════════

# class HTTPReranker:
#     """Реранкер через HTTP API. Поддерживает два формата:

#     - vLLM (Cohere-style):
#         POST {api_base}/v1/rerank
#         body: {"model": ..., "query": ..., "documents": [...], "top_n": N}
#         response: {"results": [{"index": 0, "relevance_score": 0.99}, ...]}

#     - TEI:
#         POST {api_base}/rerank
#         body: {"query": ..., "texts": [...]}
#         response: [{"index": 0, "score": 0.99}, ...]

#     Использование:
#         reranker = HTTPReranker("http://localhost:8082", "qwen3-rerank")
#         top_nodes = reranker.rerank(query, candidate_nodes)

#     backend:
#       - "vllm_score": vLLM с --task score (POST /v1/score)
#       - "vllm_rerank": vLLM Cohere-style (POST /v1/rerank)
#       - "tei": TEI (POST /rerank)
#     """

#     def __init__(self, api_base, model_name, top_n=5, backend="vllm_score"):
#         self.api_base = api_base.rstrip("/")
#         self.model_name = model_name
#         self.top_n = top_n
#         self.backend = backend

#     def rerank(self, query, nodes):
#         if not nodes:
#             return nodes

#         texts = [n.get_content() for n in nodes]

#         if self.backend == "vllm_score":
#             url = f"{self.api_base}/v1/score"
#             payload = {
#                 "model": self.model_name,
#                 "text_1": query,
#                 "text_2": texts,
#             }
#             data = requests.post(url, json=payload, timeout=60).json()
#             # vLLM возвращает: {"data": [{"index": 0, "score": 0.95}, ...]}
#             # НЕ отсортировано — сортируем сами по убыванию score
#             results = sorted(data["data"], key=lambda x: x["score"], reverse=True)
#             score_key = "score"

#         elif self.backend == "vllm_rerank":
#             url = f"{self.api_base}/v1/rerank"
#             payload = {
#                 "model": self.model_name,
#                 "query": query,
#                 "documents": texts,
#                 "top_n": len(texts),
#             }
#             data = requests.post(url, json=payload, timeout=60).json()
#             results = data.get("results", [])
#             score_key = "relevance_score"

#         elif self.backend == "tei":
#             url = f"{self.api_base}/rerank"
#             payload = {"query": query, "texts": texts}
#             results = requests.post(url, json=payload, timeout=60).json()
#             score_key = "score"

#         else:
#             raise ValueError(f"Unknown backend: {self.backend}")

#         reranked = []
#         for item in results[:self.top_n]:
#             node = nodes[item["index"]]
#             node.score = float(item[score_key])
#             reranked.append(node)
#         return reranked

#     # def __init__(self, api_base: str, model_name: str,
#     #              top_n: int = 5, backend: str = "vllm"):
#     #     assert backend in ("vllm", "tei"), f"Неизвестный backend: {backend}"
#     #     self.api_base = api_base.rstrip("/")
#     #     self.model_name = model_name
#     #     self.top_n = top_n
#     #     self.backend = backend

#     # def rerank(self, query: str, nodes: list) -> list:
#     #     """Принимает список NodeWithScore, возвращает top_n после реранкинга."""
#     #     if not nodes:
#     #         return nodes

#     #     texts = [n.get_content() for n in nodes]

#     #     if self.backend == "vllm":
#     #         url = f"{self.api_base}/v1/rerank"
#     #         payload = {
#     #             "model": self.model_name,
#     #             "query": query,
#     #             "documents": texts,
#     #             "top_n": len(texts),  # просим вернуть всё, отсортированное
#     #         }
#     #         data = requests.post(url, json=payload, timeout=60).json()
#     #         results = data.get("results", [])
#     #         score_key = "relevance_score"
#     #     else:
#     #         url = f"{self.api_base}/rerank"
#     #         payload = {"query": query, "texts": texts}
#     #         results = requests.post(url, json=payload, timeout=60).json()
#     #         score_key = "score"

#     #     # Возвращаем top_n с обновлёнными score
#     #     reranked = []
#     #     for item in results[:self.top_n]:
#     #         node = nodes[item["index"]]
#     #         node.score = float(item[score_key])
#     #         reranked.append(node)
#     #     return reranked


# def build_reranker(cfg: RerankConfig) -> Optional[HTTPReranker]:
#     """Возвращает HTTPReranker, либо None если реранкинг отключён."""
#     if not cfg.enabled:
#         return None
#     return HTTPReranker(
#         api_base=cfg.api_base,
#         model_name=cfg.api_model_name,
#         top_n=cfg.top_n,
#         backend=cfg.backend,
#     )


# ═══════════════════════════════════════════════════════════════════
# 8. МЕТРИКИ (без изменений из rag_exp.py)
# ═══════════════════════════════════════════════════════════════════

class RetrievalMetrics:
    """Метрики качества retrieval по покрытию страниц."""
    NAMES = ["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr@k"]

    @staticmethod
    def calculate(nodes, relevant_pages, k=5):
        rel = set(relevant_pages)
        found, cnt, rank = set(), 0, 0
        for i, n in enumerate(nodes[:k]):
            pages = set(n.metadata.get("pages_covered", []))
            found.update(pages)
            if pages & rel:
                cnt += 1
                if not rank:
                    rank = i + 1
        p = cnt / k if k else 0
        r = len(found & rel) / len(rel) if rel else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        return dict(zip(
            RetrievalMetrics.NAMES,
            [round(v, 4) for v in [p, r, f, float(cnt > 0), 1 / rank if rank else 0]]
        ))

    @staticmethod
    def aggregate(ms):
        return pd.DataFrame(ms).mean().round(4)


# ═══════════════════════════════════════════════════════════════════
# 9. ЗАПУСК ЭКСПЕРИМЕНТОВ
# ═══════════════════════════════════════════════════════════════════

def _index_documents(cfg: RAGConfig,
                     page_mapper: PageMapper,
                     splitter_factory: SplitterFactory,
                     client: QdrantClient):
    """Индексирует документы (или переиспользует коллекцию).
    Возвращает (index, nodes) — nodes нужны для BM25/hybrid.
    """
    setup_embedding(cfg.embedding)
    col = cfg.collection_name
    chunking = cfg.chunking

    # Создаём чанки в памяти всегда (нужны для BM25 и для метрик)
    nodes = splitter_factory.create(chunking).get_nodes_from_documents(
        [page_mapper.document]
    )
    nodes = page_mapper.enrich_nodes(nodes)
    for n in nodes:
        n.metadata.update({
            "chunk_size": chunking.chunk_size,
            "chunk_overlap": chunking.chunk_overlap,
            "splitter_type": chunking.splitter_type,
        })

    # Если коллекция уже есть и пересоздавать не нужно — открываем
    if not cfg.recreate and client.collection_exists(col):
        logger.info(f"{col} уже существует, переиспользуем")
        index = build_index(client, col, nodes=None, recreate=False)
        return index, nodes

    # Иначе индексируем с нуля
    index = build_index(client, col, nodes=nodes, recreate=cfg.recreate)
    client.update_collection(collection_name=col, metadata={
        "experiment": cfg.name,
        "chunk_size": chunking.chunk_size,
        "chunk_overlap": chunking.chunk_overlap,
        "splitter_type": chunking.splitter_type,
        "embed_model": cfg.embedding.model_name,
        "num_nodes": len(nodes),
    })
    return index, nodes


def run_experiment(cfg: RAGConfig,
                   page_mapper: PageMapper,
                   splitter_factory: SplitterFactory,
                   eval_data: List[Dict],
                   client: QdrantClient) -> pd.DataFrame:
    """Прогоняет один эксперимент и возвращает DataFrame с метриками."""
    logger.info(f"\n{'='*60}\nЭксперимент: {cfg.name}\n{'='*60}")

    # 1. Индексация
    index, nodes = _index_documents(cfg, page_mapper, splitter_factory, client)

    # 2. Сборка пайплайна retrieval
    retriever = build_retriever(index, cfg.retrieval, nodes=nodes)
    reranker = build_reranker(cfg.rerank)

    # 3. Поиск по всем запросам (синхронно)
    all_nodes = []
    t0 = time.time()
    for item in tqdm(eval_data, desc=f"Retrieve [{cfg.name}]"):
        candidates = retriever.retrieve(item["query"])
        if reranker is not None:
            candidates = reranker.rerank(item["query"], candidates)
        all_nodes.append(candidates)
    elapsed = time.time() - t0
    logger.info(f"Retrieval+rerank: {elapsed:.1f}s "
                f"({elapsed/len(eval_data):.2f}s на запрос)")

    # 4. Метрики для всех значений k
    rows = []
    for k in cfg.retrieval.top_k_values:
        ms = [RetrievalMetrics.calculate(n, item["relevant_pages"], k=k)
              for n, item in zip(all_nodes, eval_data)]
        avg = RetrievalMetrics.aggregate(ms).to_dict()
        avg.update({
            "experiment": cfg.name,
            "collection": cfg.collection_name,
            "k": k,
            "chunk_size": cfg.chunking.chunk_size,
            "chunk_overlap": cfg.chunking.chunk_overlap,
            "splitter_type": cfg.chunking.splitter_type,
            "retrieval_mode": cfg.retrieval.mode,
            "rerank": cfg.rerank.enabled,
        })
        rows.append(avg)
        logger.info(
            f"  {cfg.name} | k={k:2d} | "
            f"P={avg['precision@k']:.3f} R={avg['recall@k']:.3f} "
            f"F1={avg['f1@k']:.3f} HR={avg['hit_rate@k']:.3f} MRR={avg['mrr@k']:.3f}"
        )
    return pd.DataFrame(rows)


def run_experiments(configs: List[RAGConfig],
                    page_mapper: PageMapper,
                    splitter_factory: SplitterFactory,
                    eval_data: List[Dict],
                    qdrant_url: str = "http://127.0.0.1:6333",
                    save_csv: Optional[str] = None) -> pd.DataFrame:
    """Запускает серию экспериментов и собирает все метрики в один DataFrame.
    Используется один общий Qdrant-клиент с gRPC для скорости.
    """
    client = QdrantClient(url=qdrant_url, prefer_grpc=True)

    dfs = []
    for cfg in configs:
        df = run_experiment(cfg, page_mapper, splitter_factory, eval_data, client)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_csv, index=False)
        logger.info(f"Сохранено: {save_csv}")

    return result


# ═══════════════════════════════════════════════════════════════════
# 10. ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════

class ResultsVisualizer:
    """Графики результатов экспериментов."""
    METRICS = ["recall@k", "f1@k", "mrr@k"]

    @staticmethod
    def plot_by_variable(df, x_col, hue_col=None,
                         title="RAG Results", save_path=None, k_values=None):
        """Линейные графики — для числовых осей (chunk_size, overlap, retrieval_k)."""
        sns.set_theme(style="whitegrid")
        M = ResultsVisualizer.METRICS
        ks = k_values or sorted(df["k"].unique())

        fig, axes = plt.subplots(
            len(M), len(ks),
            figsize=(5 * len(ks), 3.5 * len(M)),
            sharex="col", sharey="row",
        )
        plt.subplots_adjust(hspace=0.3, wspace=0.15)
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

        for ri, m in enumerate(M):
            for ci, kv in enumerate(ks):
                ax = axes[ri, ci] if len(M) > 1 else axes[ci]
                sub = df[df["k"] == kv]

                kw = dict(data=sub, x=x_col, y=m, marker="o", ax=ax)
                if hue_col and hue_col in df.columns:
                    kw.update(hue=hue_col, palette="viridis")
                sns.lineplot(**kw)

                if ri == 0:
                    ax.set_title(f"K = {kv}", fontsize=14, fontweight="bold")
                ax.set_ylabel(
                    m.replace("@k", "").upper() if ci == 0 else "",
                    fontsize=12, fontweight="bold",
                )
                ax.set_xlabel(x_col if ri == len(M) - 1 else "")

                leg = ax.get_legend()
                if leg:
                    if ri == 0 and ci == len(ks) - 1:
                        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1))
                    else:
                        leg.remove()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"График: {save_path}")
        plt.show()

    @staticmethod
    def compare_categories(df, category_col="splitter_type", label_col="experiment",
                           title="Сравнение", save_path=None, k_values=None):
        """Bar chart — для категориальных сравнений (модели, splitter, retrieval mode)."""
        sns.set_theme(style="whitegrid")
        M = ResultsVisualizer.METRICS
        ks = k_values or sorted(df["k"].unique())
        cats = df[category_col].unique()
        pal = dict(zip(cats, sns.color_palette("husl", len(cats))))

        fig, axes = plt.subplots(
            len(M), len(ks),
            figsize=(4.5 * len(ks), 3.2 * len(M)),
            sharey="row",
        )
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

        for ri, m in enumerate(M):
            for ci, kv in enumerate(ks):
                ax = axes[ri, ci] if len(M) > 1 else axes[ci]
                sub = df[df["k"] == kv]

                sns.barplot(
                    data=sub, x=label_col, y=m, hue=category_col,
                    palette=pal, ax=ax, dodge=False,
                    edgecolor="white", linewidth=.5,
                )
                for c in ax.containers:
                    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)

                if ri == 0:
                    ax.set_title(f"K = {kv}", fontsize=13, fontweight="bold")
                ax.set_ylabel(
                    m.replace("@k", "").upper() if ci == 0 else "",
                    fontsize=11, fontweight="bold",
                )
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=35, labelsize=8)

                leg = ax.get_legend()
                if leg:
                    if ri == 0 and ci == len(ks) - 1:
                        ax.legend(
                            title=category_col, bbox_to_anchor=(1.05, 1),
                            fontsize=9, title_fontsize=10,
                        )
                    else:
                        leg.remove()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"График: {save_path}")
        plt.show()
