"""
RAG Experiment Framework
========================
Фреймворк для проведения экспериментов с RAG-системой.
Позволяет варьировать: chunk_size, overlap, тип сплиттера,
реранкер, LLM, embedding-модель.

Автор: Юрий (ВКР, магистратура ИИ)
Стек: LlamaIndex, Qdrant, vLLM/TEI, Python 3.12

Пример использования:
    experiments = [
        RAGConfig(name="token_500_200", chunking=ChunkingConfig(chunk_size=500, chunk_overlap=200)),
        RAGConfig(name="sentence_500_200", chunking=ChunkingConfig(chunk_size=500, chunk_overlap=200, splitter_type="sentence")),
    ]
    
    runner = ExperimentRunner(docs=docs, full_text=full_text, page_ranges=page_ranges, eval_data=eval_data)
    await runner.run_all(experiments)
"""

import json
import asyncio
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from llama_index.core import (
    VectorStoreIndex, Document, Settings, StorageContext
)
from llama_index.core.node_parser import (
    SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser,
    HierarchicalNodeParser
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from qdrant_client import QdrantClient, AsyncQdrantClient
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("rag_experiment")


# ═════════════════════════════════════════════════════════════
# 1. КОНФИГИ (DATACLASSES)
# ═════════════════════════════════════════════════════════════
#
# Каждый аспект RAG-пайплайна описывается своим конфигом.
# RAGConfig объединяет их все в один объект.
# Для нового эксперимента достаточно изменить нужные поля.
# ═════════════════════════════════════════════════════════════

@dataclass
class ChunkingConfig:
    """
    Конфигурация разбиения текста на чанки.
    
    Поля:
        chunk_size:     размер чанка в токенах (для token/sentence сплиттеров)
        chunk_overlap:  перекрытие между чанками в токенах
        splitter_type:  тип сплиттера:
                        - "token"     → TokenTextSplitter (режет строго по токенам)
                        - "sentence"  → SentenceSplitter (старается не рвать предложения)
                        - "semantic"  → SemanticSplitterNodeParser (режет по смысловым границам)
                        - "hierarchical" → HierarchicalNodeParser (иерархия чанков)
        
        # Параметры для SemanticSplitter:
        breakpoint_percentile: порог для определения смысловой границы (0-100).
                               Чем выше — тем реже разрезает (крупнее чанки).
        
        # Параметры для HierarchicalNodeParser:
        hierarchy_sizes: размеры чанков на каждом уровне иерархии.
                         Например [1024, 512, 128] — три уровня.
    """
    chunk_size: int = 500
    chunk_overlap: int = 200
    splitter_type: str = "token"  # "token" | "sentence" | "semantic" | "hierarchical"
    breakpoint_percentile: int = 90
    hierarchy_sizes: List[int] = field(default_factory=lambda: [1024, 512, 128])


@dataclass
class EmbeddingConfig:
    """
    Конфигурация embedding-модели.
    
    Поля:
        model_name:      имя модели на HuggingFace (для токенизатора и логов)
        api_base:        URL сервера эмбеддингов (TEI или vLLM)
        api_model_name:  имя модели как оно зарегистрировано на сервере
                         (может отличаться от model_name)
    """
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    api_base: str = "http://localhost:8081/v1"
    api_model_name: str = "qwen3-embed"


@dataclass
class RetrievalConfig:
    """
    Конфигурация поиска (retrieval).
    
    Поля:
        top_k_values:  список значений k, при которых считаем метрики.
                       Например [1, 2, 3, 5, 10] — считаем precision@1, precision@2 и т.д.
        retrieval_k:   сколько чанков запрашиваем у retriever (верх «воронки»).
                       Должен быть >= max(top_k_values).
        use_reranker:  использовать ли реранкер после retrieval.
        reranker_model: имя модели реранкера (если use_reranker=True).
        reranker_top_n: сколько чанков оставить после реранкинга.
    """
    top_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    retrieval_k: int = 12
    use_reranker: bool = False
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_top_n: int = 5


@dataclass
class LLMConfig:
    """
    Конфигурация LLM для генерации ответов (E2E оценка).
    На этапе retrieval-экспериментов не используется.
    
    Поля:
        model_name:     имя модели
        api_base:       URL vLLM-сервера
        api_model_name: имя модели на сервере
        max_tokens:     максимум токенов в ответе
        temperature:    температура генерации
    """
    model_name: str = "Qwen/Qwen3-4B"
    api_base: str = "http://localhost:8000/v1"
    api_model_name: str = "qwen3-4b"
    max_tokens: int = 512
    temperature: float = 0.3


@dataclass
class RAGConfig:
    """
    Главный конфиг эксперимента. Объединяет все остальные конфиги.
    
    Поля:
        name:          уникальное имя эксперимента (используется для имени коллекции,
                       файлов результатов, логов)
        chunking:      настройки чанкирования
        embedding:     настройки embedding-модели
        retrieval:     настройки поиска
        llm:           настройки LLM
        qdrant_url:    URL Qdrant-сервера
        results_dir:   папка для сохранения результатов
        recreate:      пересоздать коллекцию, если она уже существует
    """
    name: str = "baseline"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    qdrant_url: str = "http://127.0.0.1:6333"
    results_dir: str = "../results"
    recreate: bool = True

    @property
    def collection_name(self) -> str:
        """Имя коллекции в Qdrant, формируется автоматически из name."""
        return f"wb_{self.name}"


# ═════════════════════════════════════════════════════════════
# 2. РАБОТА СО СТРАНИЦАМИ (PAGE MAPPING)
# ═════════════════════════════════════════════════════════════
#
# Вместо маркеров <<<PAGE:N>>> используем символьные позиции.
# Строим карту: «от какого символа до какого идёт страница N».
# После чанкирования определяем страницы каждого чанка
# по его позиции в исходном тексте.
# ═════════════════════════════════════════════════════════════

class PageMapper:
    """
    Строит и хранит карту соответствия «символьная позиция → номер страницы».
    Используется для точной привязки чанков к страницам документа.
    """

    def __init__(self, docs: List[Document]):
        """
        Args:
            docs: список Document из LlamaIndex (каждый = одна страница PDF).
                  У каждого должен быть metadata["page_label"].
        """
        self.full_text, self.page_ranges = self._build_map(docs)
        self.document = Document(text=self.full_text)

    @staticmethod
    def _build_map(docs: List[Document]):
        """
        Объединяет все документы в один текст и строит карту позиций.
        
        Returns:
            full_text:   весь текст одной строкой
            page_ranges: список кортежей (start_char, end_char, page_label_str)
        """
        full_text = ""
        page_ranges = []
        for doc in docs:
            start = len(full_text)
            full_text += doc.text + "\n\n"
            end = len(full_text)
            page_label = str(doc.metadata["page_label"])
            page_ranges.append((start, end, page_label))
        return full_text, page_ranges

    def get_pages_for_chunk(self, chunk_start: int, chunk_end: int) -> List[str]:
        """
        Определяет, на какие страницы попадает чанк по его символьным позициям.
        
        Args:
            chunk_start: начальная позиция чанка в full_text
            chunk_end:   конечная позиция чанка в full_text
        
        Returns:
            Список номеров страниц (строки), которые покрывает этот чанк.
        """
        pages = []
        for pr_start, pr_end, page_label in self.page_ranges:
            if chunk_start < pr_end and chunk_end > pr_start:
                pages.append(page_label)
        return pages

    def enrich_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        Привязывает страницы к каждому чанку через поиск его позиции
        в исходном тексте.
        
        Args:
            nodes: список чанков (TextNode) после сплиттера
        
        Returns:
            Те же nodes с добавленным metadata["pages_covered"]
        """
        search_start = 0
        for node in nodes:
            idx = self.full_text.find(node.text, search_start)
            if idx == -1:
                idx = self.full_text.find(node.text)

            if idx != -1:
                chunk_start = idx
                chunk_end = idx + len(node.text)
                node.metadata["pages_covered"] = self.get_pages_for_chunk(
                    chunk_start, chunk_end
                )
                search_start = idx
            else:
                logger.warning(f"Чанк не найден в тексте: {node.text[:80]}...")
                node.metadata["pages_covered"] = []
        return nodes


# ═════════════════════════════════════════════════════════════
# 3. ФАБРИКА СПЛИТТЕРОВ
# ═════════════════════════════════════════════════════════════
#
# Создаёт нужный сплиттер по типу из ChunkingConfig.
# Инкапсулирует логику выбора — остальной код не знает,
# какой конкретно сплиттер используется.
# ═════════════════════════════════════════════════════════════

class SplitterFactory:
    """
    Фабрика для создания сплиттеров по конфигу.
    Загружает токенизатор один раз и переиспользует.
    """

    def __init__(self, embed_model_name: str):
        """
        Args:
            embed_model_name: имя HuggingFace модели для токенизатора.
                              Используется в TokenTextSplitter и SentenceSplitter,
                              чтобы размер чанка в токенах соответствовал
                              токенизатору embedding-модели.
        """
        logger.info(f"Загрузка токенизатора: {embed_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

    def create(self, config: ChunkingConfig) -> Any:
        """
        Создаёт сплиттер по конфигу.
        
        Args:
            config: ChunkingConfig с параметрами
        
        Returns:
            Объект сплиттера LlamaIndex
        
        Raises:
            ValueError: если splitter_type неизвестен
        """
        if config.splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                tokenizer=self._tokenizer.encode,
            )

        elif config.splitter_type == "sentence":
            return SentenceSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                tokenizer=self._tokenizer.encode,
            )

        elif config.splitter_type == "semantic":
            # SemanticSplitter использует embedding-модель для определения
            # смысловых границ. Размер чанков определяется автоматически.
            # chunk_size и chunk_overlap НЕ используются.
            embed_model = Settings.embed_model
            if embed_model is None:
                raise RuntimeError(
                    "Для SemanticSplitter нужна embed_model в Settings. "
                    "Вызовите setup_embedding() перед созданием сплиттера."
                )
            return SemanticSplitterNodeParser(
                embed_model=embed_model,
                breakpoint_percentile_threshold=config.breakpoint_percentile,
            )

        elif config.splitter_type == "hierarchical":
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=config.hierarchy_sizes,
            )

        else:
            raise ValueError(
                f"Неизвестный тип сплиттера: {config.splitter_type}. "
                f"Допустимые: token, sentence, semantic, hierarchical"
            )


# ═════════════════════════════════════════════════════════════
# 4. МЕТРИКИ
# ═════════════════════════════════════════════════════════════
#
# Все метрики retrieval-качества в одном классе.
# Считает precision, recall, f1, hit_rate, mrr для одного запроса.
# Агрегирует по набору запросов.
# ═════════════════════════════════════════════════════════════

class RetrievalMetrics:
    """
    Подсчёт метрик качества retrieval.
    Все метрики бинарные на уровне чанков: чанк считается релевантным,
    если хотя бы одна из его страниц совпадает с релевантными страницами вопроса.
    """

    @staticmethod
    def calculate(
        retrieved_nodes: List[Any],
        relevant_pages: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Метрики для ОДНОГО запроса.
        
        Args:
            retrieved_nodes: чанки, возвращённые retriever-ом
            relevant_pages:  список номеров страниц с правильным ответом
            k:               top-k для расчёта метрик
        
        Returns:
            Словарь с метриками, все значения округлены до 4 знаков:
            {
                "precision@k": доля релевантных чанков в top-k,
                "recall@k":    доля найденных релевантных страниц,
                "f1@k":        гармоническое среднее precision и recall,
                "hit_rate@k":  1.0 если хотя бы один чанк релевантен,
                "mrr@k":       1 / ранг первого релевантного чанка
            }
        """
        relevant_set = set(relevant_pages)
        retrieved_pages = set()
        relevant_chunks_count = 0
        first_relevant_rank = 0

        for i, node in enumerate(retrieved_nodes[:k]):
            chunk_pages = set(node.metadata.get("pages_covered", []))
            overlap = chunk_pages & relevant_set
            retrieved_pages.update(chunk_pages)

            if overlap:
                relevant_chunks_count += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = i + 1

        precision = relevant_chunks_count / k if k > 0 else 0.0
        recall = (
            len(retrieved_pages & relevant_set) / len(relevant_set)
            if relevant_set else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        hit_rate = 1.0 if relevant_chunks_count > 0 else 0.0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0

        return {
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "f1@k": round(f1, 4),
            "hit_rate@k": round(hit_rate, 4),
            "mrr@k": round(mrr, 4),
        }

    @staticmethod
    def aggregate(all_metrics: List[Dict[str, float]]) -> pd.Series:
        """
        Агрегация метрик по всем запросам (среднее).
        
        Args:
            all_metrics: список словарей от calculate()
        
        Returns:
            pd.Series со средними значениями
        """
        return pd.DataFrame(all_metrics).mean().round(4)


# ═════════════════════════════════════════════════════════════
# 5. ЗАГРУЗКА EVAL-ДАННЫХ
# ═════════════════════════════════════════════════════════════

class EvalDataLoader:
    """
    Загрузка тестовых данных из JSONL-файла.
    
    Формат JSONL:
        {"doc_1": {"simple": "...", "medium": "...", "complex": "..."}}
        {"doc_5": {"simple": "...", "medium": "...", "complex": "..."}}
    
    Ключ "doc_N" определяет номер страницы → relevant_pages = ["N"].
    Если ключ "doc_5_12" → relevant_pages = ["5", "12"].
    """

    @staticmethod
    def load(
        path: str,
        question_types: List[str] = None,
        skip_pages: set = None
    ) -> List[Dict]:
        """
        Args:
            path:           путь к JSONL-файлу
            question_types: какие типы вопросов загружать.
                            По умолчанию ["simple", "medium", "complex"] — все.
                            Можно указать ["simple"] для загрузки только простых.
            skip_pages:     множество номеров страниц, которые нужно пропустить.
                            Например {72, 73, 74, ...} для справочных таблиц.
        
        Returns:
            Список словарей: [{"query": str, "relevant_pages": List[str]}, ...]
        """
        if question_types is None:
            question_types = ["simple", "medium", "complex"]
        if skip_pages is None:
            skip_pages = set()

        eval_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                key = list(entry.keys())[0]          # "doc_5" или "doc_5_12"
                questions = list(entry.values())[0]   # {"simple": ..., "medium": ..., "complex": ...}
                relevant_pages = key.split("_")[1:]   # ["5"] или ["5", "12"]

                # Пропускаем страницы без фактической информации
                if any(int(p) in skip_pages for p in relevant_pages):
                    continue

                for q_type in question_types:
                    if q_type in questions and questions[q_type]:
                        eval_data.append({
                            "query": questions[q_type],
                            "relevant_pages": relevant_pages,
                            "question_type": q_type,
                        })

        logger.info(f"Загружено {len(eval_data)} вопросов из {path}")
        return eval_data


# ═════════════════════════════════════════════════════════════
# 6. ГЛАВНЫЙ КЛАСС — RAGExperiment
# ═════════════════════════════════════════════════════════════
#
# Один объект = один эксперимент с конкретными параметрами.
# Метод run() выполняет полный цикл:
#   создание коллекции → оценка → сохранение.
# ═════════════════════════════════════════════════════════════

class RAGExperiment:
    """
    Проведение одного RAG-эксперимента.
    
    Жизненный цикл:
        1. __init__()        — сохраняет конфиг, создаёт клиенты
        2. setup_embedding() — подключает embedding-модель
        3. create_collection() — чанкирует текст, строит индекс в Qdrant
        4. evaluate()        — прогоняет eval-данные, считает метрики
        5. run()             — объединяет шаги 2-4
    """

    def __init__(self, config: RAGConfig, page_mapper: PageMapper, splitter_factory: SplitterFactory,
                 sync_client: QdrantClient = None, async_client: AsyncQdrantClient = None):
        """
        Args:
            config:           RAGConfig с параметрами эксперимента
            page_mapper:      PageMapper для привязки страниц к чанкам
            splitter_factory: SplitterFactory для создания сплиттеров
            sync_client:      общий QdrantClient (если None — создаётся новый)
            async_client:     общий AsyncQdrantClient (если None — создаётся новый)
        """
        self.config = config
        self.page_mapper = page_mapper
        self.splitter_factory = splitter_factory

        self.sync_client = sync_client or QdrantClient(url=config.qdrant_url)
        self.async_client = async_client or AsyncQdrantClient(
            url=config.qdrant_url, prefer_grpc=True
        )

    def setup_embedding(self):
        """Подключает embedding-модель через OpenAI-совместимый API (TEI/vLLM).
        Пропускает, если модель уже настроена с теми же параметрами."""
        cfg = self.config.embedding
        current = Settings.embed_model
        # Если уже настроена та же модель — не пересоздаём
        if (current is not None
                and hasattr(current, "model_name")
                and current.model_name == cfg.api_model_name):
            return
        Settings.embed_model = OpenAILikeEmbedding(
            model_name=cfg.api_model_name,
            api_base=cfg.api_base,
            api_key="fake",
        )
        logger.info(f"Embedding: {cfg.model_name} @ {cfg.api_base}")

    def create_collection(self):
        """
        Создаёт коллекцию в Qdrant:
        1. Создаёт сплиттер по конфигу
        2. Разбивает текст на чанки
        3. Привязывает страницы к чанкам
        4. Добавляет метаинформацию
        5. Строит индекс и загружает в Qdrant
        """
        col_name = self.config.collection_name
        cfg = self.config.chunking

        # Удаляем старую коллекцию если нужно
        if self.config.recreate and self.sync_client.collection_exists(col_name):
            self.sync_client.delete_collection(col_name)
            logger.info(f"Удалена старая коллекция: {col_name}")

        # Если коллекция уже есть и recreate=False — пропускаем
        if self.sync_client.collection_exists(col_name):
            logger.info(f"Коллекция {col_name} уже существует, пропускаем создание")
            return

        # Создаём сплиттер и разбиваем текст
        splitter = self.splitter_factory.create(cfg)
        nodes = splitter.get_nodes_from_documents([self.page_mapper.document])
        logger.info(f"Создано {len(nodes)} чанков (splitter={cfg.splitter_type}, "
                     f"size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

        # Привязываем страницы
        nodes = self.page_mapper.enrich_nodes(nodes)

        # Добавляем метаинформацию в каждый чанк
        for node in nodes:
            node.metadata["chunk_size"] = cfg.chunk_size
            node.metadata["chunk_overlap"] = cfg.chunk_overlap
            node.metadata["splitter_type"] = cfg.splitter_type
            node.metadata["experiment"] = self.config.name

        # Создаём индекс в Qdrant
        vector_store = QdrantVectorStore(
            collection_name=col_name,
            client=self.sync_client,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        # Сохраняем метаданные коллекции
        self.sync_client.update_collection(
            collection_name=col_name,
            metadata={
                "experiment_name": self.config.name,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "splitter_type": cfg.splitter_type,
                "embed_model": self.config.embedding.model_name,
                "num_nodes": len(nodes),
            }
        )
        logger.info(f"Коллекция {col_name} создана: {len(nodes)} чанков")

    async def evaluate(self, eval_data: List[Dict]) -> pd.DataFrame:
        """
        Оценка качества retrieval по набору тестовых вопросов.
        
        Args:
            eval_data: список из EvalDataLoader.load()
        
        Returns:
            pd.DataFrame с результатами:
            одна строка на каждую комбинацию (эксперимент, k).
        """
        col_name = self.config.collection_name
        retrieval_cfg = self.config.retrieval

        # Подключаемся к коллекции через асинхронный клиент
        vector_store = QdrantVectorStore(
            collection_name=col_name,
            aclient=self.async_client,
        )
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = index.as_retriever(
            similarity_top_k=retrieval_cfg.retrieval_k
        )

        # Запускаем запросы с ограничением параллелизма.
        # Без семафора 222 запроса уходят на TEI одновременно,
        # TEI ставит их в очередь и обрабатывает медленно.
        semaphore = asyncio.Semaphore(20)  # макс 20 одновременных запросов

        async def _retrieve_one(query: str):
            async with semaphore:
                return await retriever.aretrieve(query)

        tasks = [_retrieve_one(item["query"]) for item in eval_data]
        all_retrieved = await asyncio.gather(*tasks)

        # Считаем метрики для каждого k
        results = []
        for k in retrieval_cfg.top_k_values:
            all_metrics = []
            for nodes, item in zip(all_retrieved, eval_data):
                metrics = RetrievalMetrics.calculate(
                    nodes, item["relevant_pages"], k=k
                )
                all_metrics.append(metrics)

            avg = RetrievalMetrics.aggregate(all_metrics)
            row = avg.to_dict()
            row["experiment"] = self.config.name
            row["collection"] = col_name
            row["k"] = k
            row["chunk_size"] = self.config.chunking.chunk_size
            row["chunk_overlap"] = self.config.chunking.chunk_overlap
            row["splitter_type"] = self.config.chunking.splitter_type
            results.append(row)

            logger.info(
                f"  {self.config.name} | k={k:2d} | "
                f"P={row['precision@k']:.3f} R={row['recall@k']:.3f} "
                f"F1={row['f1@k']:.3f} HR={row['hit_rate@k']:.3f} "
                f"MRR={row['mrr@k']:.3f}"
            )

        return pd.DataFrame(results)

    async def run(self, eval_data: List[Dict]) -> pd.DataFrame:
        """
        Полный цикл эксперимента: embedding → коллекция → оценка.
        
        Args:
            eval_data: тестовые данные
        
        Returns:
            pd.DataFrame с результатами
        """
        logger.info(f"{'='*60}")
        logger.info(f"Эксперимент: {self.config.name}")
        logger.info(f"{'='*60}")

        self.setup_embedding()
        self.create_collection()
        results = await self.evaluate(eval_data)
        return results


# ═════════════════════════════════════════════════════════════
# 7. ВИЗУАЛИЗАЦИЯ
# ═════════════════════════════════════════════════════════════

class ResultsVisualizer:
    """
    Построение графиков по результатам экспериментов.
    """

    METRICS = ["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr@k"]

    @staticmethod
    def plot_by_variable(
        df: pd.DataFrame,
        x_col: str,
        hue_col: str = None,
        title: str = "RAG Experiment Results",
        save_path: str = None,
        k_values: List[int] = None,
    ):
        """
        Универсальный метод для построения сетки графиков.
        
        Args:
            df:        DataFrame с результатами (из ExperimentRunner.run_all())
            x_col:     колонка по оси X (например "chunk_size", "chunk_overlap")
            hue_col:   колонка для цветового разделения (например "splitter_type")
            title:     заголовок всей фигуры
            save_path: путь для сохранения PNG (None = не сохранять)
            k_values:  какие значения k отображать (None = все)
        """
        sns.set_theme(style="whitegrid")
        metrics = ResultsVisualizer.METRICS

        if k_values is None:
            k_values = sorted(df["k"].unique())

        fig, axes = plt.subplots(
            len(metrics), len(k_values),
            figsize=(5 * len(k_values), 3.5 * len(metrics)),
            sharex="col", sharey="row",
        )
        plt.subplots_adjust(hspace=0.3, wspace=0.15)
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

        for row_idx, metric in enumerate(metrics):
            for col_idx, k_val in enumerate(k_values):
                ax = axes[row_idx, col_idx] if len(metrics) > 1 else axes[col_idx]
                subset = df[df["k"] == k_val]

                plot_kwargs = dict(
                    data=subset, x=x_col, y=metric,
                    marker="o", ax=ax,
                )
                if hue_col and hue_col in df.columns:
                    plot_kwargs["hue"] = hue_col
                    plot_kwargs["palette"] = "viridis"

                sns.lineplot(**plot_kwargs)

                # Заголовки и подписи
                if row_idx == 0:
                    ax.set_title(f"K = {k_val}", fontsize=14, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        metric.replace("@k", "").upper(),
                        fontsize=12, fontweight="bold",
                    )
                else:
                    ax.set_ylabel("")
                ax.set_xlabel(x_col if row_idx == len(metrics) - 1 else "")

                # Легенда только в правом верхнем углу
                legend = ax.get_legend()
                if legend:
                    if row_idx == 0 and col_idx == len(k_values) - 1:
                        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1))
                    else:
                        legend.remove()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"График сохранён: {save_path}")
        plt.show()

    @staticmethod
    def compare_categories(
        df: pd.DataFrame,
        category_col: str = "splitter_type",
        label_col: str = "experiment",
        title: str = "Сравнение конфигураций",
        save_path: str = None,
        k_values: List[int] = None,
    ):
        """
        Bar chart для сравнения категориальных переменных (сплиттеры, модели и т.д.).
        Каждая категория — отдельный столбец, сгруппированный по k.

        Args:
            df:            DataFrame с результатами
            category_col:  колонка с категорией для цвета ("splitter_type", "experiment")
            label_col:     колонка для подписей на оси X ("experiment")
            title:         заголовок
            save_path:     путь для сохранения PNG
            k_values:      какие значения k отображать (None = все)
        """
        sns.set_theme(style="whitegrid")
        metrics = ResultsVisualizer.METRICS

        if k_values is None:
            k_values = sorted(df["k"].unique())

        fig, axes = plt.subplots(
            len(metrics), len(k_values),
            figsize=(4.5 * len(k_values), 3.2 * len(metrics)),
            sharey="row",
        )
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

        # Палитра по категориям
        categories = df[category_col].unique()
        palette = dict(zip(categories, sns.color_palette("husl", len(categories))))

        for row_idx, metric in enumerate(metrics):
            for col_idx, k_val in enumerate(k_values):
                ax = axes[row_idx, col_idx] if len(metrics) > 1 else axes[col_idx]
                subset = df[df["k"] == k_val].copy()

                sns.barplot(
                    data=subset,
                    x=label_col,
                    y=metric,
                    hue=category_col,
                    palette=palette,
                    ax=ax,
                    dodge=False,
                    edgecolor="white",
                    linewidth=0.5,
                )

                # Значения над столбцами
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

                # Заголовки
                if row_idx == 0:
                    ax.set_title(f"K = {k_val}", fontsize=13, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(
                        metric.replace("@k", "").upper(),
                        fontsize=11, fontweight="bold",
                    )
                else:
                    ax.set_ylabel("")

                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=35, labelsize=8)

                # Легенда только один раз
                legend = ax.get_legend()
                if legend:
                    if row_idx == 0 and col_idx == len(k_values) - 1:
                        ax.legend(
                            title=category_col, bbox_to_anchor=(1.05, 1),
                            fontsize=9, title_fontsize=10,
                        )
                    else:
                        legend.remove()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"График сохранён: {save_path}")
        plt.show()

    @staticmethod
    def summary_table(
        df: pd.DataFrame,
        group_col: str = "experiment",
        k_values: List[int] = None,
        save_path: str = None,
    ) -> pd.DataFrame:
        """
        Сводная таблица метрик по экспериментам.

        Args:
            df:        DataFrame с результатами
            group_col: колонка для группировки строк ("experiment", "splitter_type")
            k_values:  какие k включить (None = все)
            save_path: путь для сохранения CSV (None = не сохранять)

        Returns:
            pd.DataFrame — сводная таблица с MultiIndex (group_col, k)
        """
        metrics = ResultsVisualizer.METRICS

        if k_values is None:
            k_values = sorted(df["k"].unique())

        subset = df[df["k"].isin(k_values)]
        pivot = subset.pivot_table(
            index=[group_col, "k"],
            values=metrics,
            aggfunc="mean",
        ).round(4)

        # Переупорядочим колонки
        pivot = pivot[metrics]

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            pivot.to_csv(save_path)
            logger.info(f"Таблица сохранена: {save_path}")

        return pivot


# ═════════════════════════════════════════════════════════════
# 8. ОРКЕСТРАТОР — ExperimentRunner
# ═════════════════════════════════════════════════════════════
#
# Запускает серию экспериментов, собирает результаты,
# сохраняет CSV и строит графики.
# ═════════════════════════════════════════════════════════════

class ExperimentRunner:
    """
    Оркестратор для запуска серии RAG-экспериментов.
    
    Принимает список RAGConfig, для каждого создаёт RAGExperiment,
    запускает его, собирает все результаты в один DataFrame,
    сохраняет CSV и строит графики.
    """

    def __init__(
        self,
        docs: List[Document],
        eval_data: List[Dict],
        embed_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        qdrant_url: str = "http://127.0.0.1:6333",
    ):
        """
        Args:
            docs:             список Document (страницы PDF, с page_label)
            eval_data:        тестовые данные из EvalDataLoader.load()
            embed_model_name: имя модели для токенизатора сплиттера
            qdrant_url:       URL Qdrant-сервера
        """
        self.page_mapper = PageMapper(docs)
        self.splitter_factory = SplitterFactory(embed_model_name)
        self.eval_data = eval_data
        self.all_results = pd.DataFrame()

        # Общие клиенты Qdrant — переиспользуются между экспериментами
        self._sync_client = QdrantClient(url=qdrant_url)
        self._async_client = AsyncQdrantClient(url=qdrant_url, prefer_grpc=True)

    async def run_all(
        self,
        configs: List[RAGConfig],
        save_csv: str = None,
    ) -> pd.DataFrame:
        """
        Запускает все эксперименты последовательно.
        """
        all_dfs = []

        for cfg in configs:
            experiment = RAGExperiment(
                config=cfg,
                page_mapper=self.page_mapper,
                splitter_factory=self.splitter_factory,
                sync_client=self._sync_client,
                async_client=self._async_client,
            )
            df = await experiment.run(self.eval_data)
            all_dfs.append(df)

        self.all_results = pd.concat(all_dfs, ignore_index=True)

        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            self.all_results.to_csv(save_csv, index=False)
            logger.info(f"Результаты сохранены: {save_csv}")

        return self.all_results

    def plot(self, x_col: str, hue_col: str = None, title: str = None, save_path: str = None):
        """
        Линейные графики — для числовых осей (chunk_size, overlap).
        """
        if self.all_results.empty:
            logger.warning("Нет результатов для визуализации")
            return
        ResultsVisualizer.plot_by_variable(
            df=self.all_results, x_col=x_col, hue_col=hue_col,
            title=title or f"RAG Results by {x_col}", save_path=save_path,
        )

    def compare(
        self, category_col: str = "splitter_type", label_col: str = "experiment",
        title: str = None, save_path: str = None, k_values: List[int] = None,
    ):
        """
        Bar chart — для категориальных сравнений (сплиттеры, модели).
        """
        if self.all_results.empty:
            logger.warning("Нет результатов для визуализации")
            return
        ResultsVisualizer.compare_categories(
            df=self.all_results, category_col=category_col,
            label_col=label_col,
            title=title or "Сравнение конфигураций", save_path=save_path,
            k_values=k_values,
        )

    def table(
        self, group_col: str = "experiment",
        k_values: List[int] = None, save_path: str = None,
    ) -> pd.DataFrame:
        """
        Сводная таблица метрик.
        """
        if self.all_results.empty:
            logger.warning("Нет результатов")
            return pd.DataFrame()
        return ResultsVisualizer.summary_table(
            df=self.all_results, group_col=group_col,
            k_values=k_values, save_path=save_path,
        )