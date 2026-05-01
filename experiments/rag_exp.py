"""
RAG Experiment Framework (v2)
=============================
Упрощённый фреймворк для экспериментов с RAG-системой.

Пример:
    from rag_experiment import *

    docs = load_and_prepare_docs("../data")
    page_mapper = PageMapper(docs)
    eval_data = EvalDataLoader.load("../context/questions.jsonl", skip_pages={72,...,81})
    splitter_factory = SplitterFactory("Qwen/Qwen3-Embedding-0.6B")

    results = await run_experiments(
        configs=[RAGConfig(name="token_500_200"), ...],
        page_mapper=page_mapper,
        splitter_factory=splitter_factory,
        eval_data=eval_data,
        save_csv="../results/exp1.csv",
    )
    ResultsVisualizer.plot_by_variable(results, x_col="chunk_size")
"""

import json, asyncio, logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from llama_index.core import (
    VectorStoreIndex, Document, Settings, StorageContext, SimpleDirectoryReader,
)
from llama_index.core.node_parser import (
    SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser, HierarchicalNodeParser,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("rag")

# ═══════════════════ КОНФИГИ ═══════════════════

@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 200
    splitter_type: str = "token"
    breakpoint_percentile: int = 90
    hierarchy_sizes: List[int] = field(default_factory=lambda: [1024, 512, 128])

@dataclass
class EmbeddingConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    api_base: str = "http://localhost:8081/v1"
    api_model_name: str = "qwen3-embed"
    params: dict = None

@dataclass
class RetrievalConfig:
    top_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    retrieval_k: int = 12
    concurrency: int = 20
    mode: str = "dense"   # ADD
    use_rerank: bool = False  # ADD
    rerank_top_n: int = 5   # Documents to keep after reranking  # ADD

@dataclass
class RAGConfig:
    name: str = "baseline"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    qdrant_url: str = "http://127.0.0.1:6333"
    recreate: bool = True
    rerank_model: str = "codefuse-ai/F2LLM-v2-0.6B" # ADD

    @property
    def collection_name(self) -> str:
        return f"wb_{self.name}"

# ═══════════════════ HELPERS ═══════════════════

def load_and_prepare_docs(data_dir: str) -> List[Document]:
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

def ensure_embedding(config: EmbeddingConfig):
    """Гарантирует, что Settings.embed_model установлена. Безопасна для повторных вызовов."""
    # logger.info(f"ОШБИКА ЗДЕСЬ1")
    if Settings._embed_model is None:
        Settings.embed_model = OpenAILikeEmbedding(
        model_name=config.api_model_name, api_base=config.api_base,
        additional_kwargs=config.params
        )
        logger.info(f"Embedding: {config.model_name} @ {config.api_base}")
    cur = Settings.embed_model
    # logger.info(f"ОШБИКА ЗДЕСЬ2")
    if cur is not None and hasattr(cur, "model_name") and cur.model_name == config.api_model_name:
        # print("DJN PLTCM")S
        return
    

# ═══════════════════ PAGE MAPPER ═══════════════════

class PageMapper:
    def __init__(self, docs: List[Document]):
        self.full_text, self.page_ranges = self._build(docs)
        self.document = Document(text=self.full_text)

    @staticmethod
    def _build(docs):
        text, ranges = "", []
        for doc in docs:
            s = len(text)
            text += doc.text + "\n\n"
            ranges.append((s, len(text), str(doc.metadata["page_label"])))
        return text, ranges

    def get_pages(self, start, end):
        return [p for s, e, p in self.page_ranges if start < e and end > s]

    def enrich_nodes(self, nodes):
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

# ═══════════════════ SPLITTER FACTORY ═══════════════════

class SplitterFactory:
    def __init__(self, embed_model_name: str):
        self._tok = AutoTokenizer.from_pretrained(embed_model_name)
        logger.info(f"Токенизатор: {embed_model_name}")

    def create(self, cfg: ChunkingConfig):
        t = cfg.splitter_type
        if t == "token":
            return TokenTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, tokenizer=self._tok.encode)
        if t == "sentence":
            return SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, tokenizer=self._tok.encode)
        if t == "semantic":
            assert Settings.embed_model, "Вызовите ensure_embedding() перед semantic splitter"
            return SemanticSplitterNodeParser(embed_model=Settings.embed_model, breakpoint_percentile_threshold=cfg.breakpoint_percentile)
        if t == "hierarchical":
            return HierarchicalNodeParser.from_defaults(chunk_sizes=cfg.hierarchy_sizes)
        raise ValueError(f"Неизвестный splitter_type: {t}")

# ═══════════════════ METRICS ═══════════════════

class RetrievalMetrics:
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
                if not rank: rank = i + 1
        p = cnt / k if k else 0
        r = len(found & rel) / len(rel) if rel else 0
        f = 2*p*r/(p+r) if (p+r) else 0
        return dict(zip(RetrievalMetrics.NAMES, [round(v,4) for v in [p, r, f, float(cnt>0), 1/rank if rank else 0]]))

    @staticmethod
    def aggregate(ms):
        return pd.DataFrame(ms).mean().round(4)

# ═══════════════════ EVAL DATA ═══════════════════

class EvalDataLoader:
    @staticmethod
    def load(path, question_types=None, skip_pages=None):
        qt = question_types or ["simple", "medium", "complex"]
        sp = skip_pages or set()
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                key, qs = list(e.keys())[0], list(e.values())[0]
                pages = key.split("_")[1:]
                if any(int(p) in sp for p in pages): continue
                for t in qt:
                    if t in qs and qs[t]:
                        data.append({"query": qs[t], "relevant_pages": pages, "question_type": t})
        logger.info(f"Загружено {len(data)} вопросов из {path}")
        return data

# ═══════════════════ RAG EXPERIMENT ═══════════════════

class RAGExperiment:
    """Один эксперимент. Самодостаточный — сам настраивает embedding перед каждой операцией."""

    def __init__(self, config: RAGConfig, page_mapper: PageMapper, splitter_factory: SplitterFactory,
                 sync_client: QdrantClient = None, async_client: AsyncQdrantClient = None):
        self.config = config
        self.page_mapper = page_mapper
        self.splitter_factory = splitter_factory
        self.sync_client = sync_client or QdrantClient(url=config.qdrant_url)
        self.async_client = async_client or AsyncQdrantClient(url=config.qdrant_url, prefer_grpc=True)

    def create_collection(self):
        logger.info(f"CREATE COLLECTION")
        ensure_embedding(self.config.embedding)
        col = self.config.collection_name
        cfg = self.config.chunking

        if self.config.recreate and self.sync_client.collection_exists(col):
            self.sync_client.delete_collection(col)
        if self.sync_client.collection_exists(col):
            logger.info(f"{col} уже существует, пропускаем")
            return

        nodes = self.splitter_factory.create(cfg).get_nodes_from_documents([self.page_mapper.document])
        nodes = self.page_mapper.enrich_nodes(nodes)
        for n in nodes:
            n.metadata.update({"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "splitter_type": cfg.splitter_type})

        vs = QdrantVectorStore(collection_name=col, client=self.sync_client)
        VectorStoreIndex(nodes=nodes, storage_context=StorageContext.from_defaults(vector_store=vs), show_progress=True)
        self.sync_client.update_collection(collection_name=col, metadata={
            "experiment": self.config.name, "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap, "splitter_type": cfg.splitter_type,
            "embed_model": self.config.embedding.model_name, "num_nodes": len(nodes),
        })
        logger.info(f"{col}: {len(nodes)} чанков ({cfg.splitter_type})")

    async def evaluate(self, eval_data: List[Dict]) -> pd.DataFrame:
        ensure_embedding(self.config.embedding)
        col = self.config.collection_name
        rcfg = self.config.retrieval

        vs = QdrantVectorStore(collection_name=col, aclient=self.async_client)
        index = VectorStoreIndex.from_vector_store(vector_store=vs)
        retriever = index.as_retriever(similarity_top_k=rcfg.retrieval_k)

        sem = asyncio.Semaphore(rcfg.concurrency)
        async def _r(q):
            async with sem:
                return await retriever.aretrieve(q)

        all_nodes = await asyncio.gather(*[_r(item["query"]) for item in eval_data])

        rows = []
        for k in rcfg.top_k_values:
            ms = [RetrievalMetrics.calculate(n, it["relevant_pages"], k=k) for n, it in zip(all_nodes, eval_data)]
            avg = RetrievalMetrics.aggregate(ms).to_dict()
            avg.update({"experiment": self.config.name, "collection": col, "k": k,
                        "chunk_size": self.config.chunking.chunk_size,
                        "chunk_overlap": self.config.chunking.chunk_overlap,
                        "splitter_type": self.config.chunking.splitter_type})
            rows.append(avg)
            logger.info(f"  {self.config.name} | k={k:2d} | P={avg['precision@k']:.3f} R={avg['recall@k']:.3f} F1={avg['f1@k']:.3f} HR={avg['hit_rate@k']:.3f} MRR={avg['mrr@k']:.3f}")
        return pd.DataFrame(rows)

    async def run(self, eval_data: List[Dict]) -> pd.DataFrame:
        logger.info(f"{'='*60}\nЭксперимент: {self.config.name}\n{'='*60}")
        self.create_collection()
        return await self.evaluate(eval_data)

# ═══════════════════ run_experiments ═══════════════════

async def run_experiments(configs, page_mapper, splitter_factory, eval_data,
                          save_csv=None, qdrant_url="http://127.0.0.1:6333"):
    """Запускает серию экспериментов, возвращает сводный DataFrame."""
    sc = QdrantClient(url=qdrant_url)
    ac = AsyncQdrantClient(url=qdrant_url, prefer_grpc=True)
    dfs = []
    for cfg in configs:
        exp = RAGExperiment(config=cfg, page_mapper=page_mapper, splitter_factory=splitter_factory,
                            sync_client=sc, async_client=ac)
        dfs.append(await exp.run(eval_data))
    result = pd.concat(dfs, ignore_index=True)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_csv, index=False)
        logger.info(f"Сохранено: {save_csv}")
    return result

# ═══════════════════ ВИЗУАЛИЗАЦИЯ ═══════════════════

class ResultsVisualizer:
    # METRICS = ["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr@k"]
    METRICS = ["recall@k", "f1@k", "mrr@k"]

    @staticmethod
    def plot_by_variable(df, x_col, hue_col=None, title="RAG Results", save_path=None, k_values=None):
        """Линейные графики — для числовых осей (chunk_size, overlap)."""
        sns.set_theme(style="whitegrid")
        M = ResultsVisualizer.METRICS
        ks = k_values or sorted(df["k"].unique())
        fig, axes = plt.subplots(len(M), len(ks), figsize=(5*len(ks), 3.5*len(M)), sharex="col", sharey="row")
        plt.subplots_adjust(hspace=0.3, wspace=0.15)
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
        for ri, m in enumerate(M):
            for ci, kv in enumerate(ks):
                ax = axes[ri, ci] if len(M) > 1 else axes[ci]
                sub = df[df["k"] == kv]
                kw = dict(data=sub, x=x_col, y=m, marker="o", ax=ax)
                if hue_col and hue_col in df.columns: kw.update(hue=hue_col, palette="viridis")
                sns.lineplot(**kw)
                if ri == 0: ax.set_title(f"K = {kv}", fontsize=14, fontweight="bold")
                ax.set_ylabel(m.replace("@k","").upper() if ci==0 else "", fontsize=12, fontweight="bold")
                ax.set_xlabel(x_col if ri==len(M)-1 else "")
                leg = ax.get_legend()
                if leg:
                    if ri==0 and ci==len(ks)-1: ax.legend(title=hue_col, bbox_to_anchor=(1.05,1))
                    else: leg.remove()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight"); logger.info(f"График: {save_path}")
        plt.show()

    @staticmethod
    def compare_categories(df, category_col="splitter_type", label_col="experiment",
                           title="Сравнение", save_path=None, k_values=None):
        """Bar chart — для категориальных сравнений (сплиттеры, модели)."""
        sns.set_theme(style="whitegrid")
        M = ResultsVisualizer.METRICS
        ks = k_values or sorted(df["k"].unique())
        cats = df[category_col].unique()
        pal = dict(zip(cats, sns.color_palette("husl", len(cats))))
        fig, axes = plt.subplots(len(M), len(ks), figsize=(4.5*len(ks), 3.2*len(M)), sharey="row")
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
        for ri, m in enumerate(M):
            for ci, kv in enumerate(ks):
                ax = axes[ri, ci] if len(M) > 1 else axes[ci]
                sub = df[df["k"] == kv]
                sns.barplot(data=sub, x=label_col, y=m, hue=category_col, palette=pal, ax=ax, dodge=False, edgecolor="white", linewidth=.5)
                for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
                if ri == 0: ax.set_title(f"K = {kv}", fontsize=13, fontweight="bold")
                ax.set_ylabel(m.replace("@k","").upper() if ci==0 else "", fontsize=11, fontweight="bold")
                ax.set_xlabel(""); ax.tick_params(axis="x", rotation=35, labelsize=8)
                leg = ax.get_legend()
                if leg:
                    if ri==0 and ci==len(ks)-1: ax.legend(title=category_col, bbox_to_anchor=(1.05,1), fontsize=9, title_fontsize=10)
                    else: leg.remove()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight"); logger.info(f"График: {save_path}")
        plt.show()

    @staticmethod
    def summary_table(df, group_col="experiment", k_values=None, save_path=None):
        """Сводная таблица метрик."""
        M = ResultsVisualizer.METRICS
        ks = k_values or sorted(df["k"].unique())
        pv = df[df["k"].isin(ks)].pivot_table(index=[group_col, "k"], values=M, aggfunc="mean").round(4)[M]
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            pv.to_csv(save_path); logger.info(f"Таблица: {save_path}")
        return pv