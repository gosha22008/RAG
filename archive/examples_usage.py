"""
Примеры использования RAG Experiment Framework
===============================================

Этот файл — шаблон для Jupyter-ноутбука.
Копируй нужные блоки в ячейки ноутбука.
"""

# ═════════════════════════════════════════════════════════════
# ЯЧЕЙКА 1: Импорты и загрузка данных
# ═════════════════════════════════════════════════════════════

from llama_index.core import SimpleDirectoryReader
from pathlib import Path

from experiments.rag_experiment import (
    RAGConfig, ChunkingConfig, EmbeddingConfig, RetrievalConfig, LLMConfig,
    EvalDataLoader, ExperimentRunner, ResultsVisualizer,
)

# Загрузка документов
local_data = Path.cwd() / "../data"
reader = SimpleDirectoryReader(input_dir=local_data, required_exts=[".pdf"])
docs = reader.load_data()

# Проставляем page_label до удаления пустых (чтобы номера совпадали с PDF)
for i, doc in enumerate(docs):
    doc.metadata["page_label"] = i + 1

# Удаляем пустые страницы
empty = [i for i, doc in enumerate(docs) if len(doc.text) == 0]
print(f"Пустые страницы: {[docs[i].metadata['page_label'] for i in empty]}")
for i in sorted(empty, reverse=True):
    docs.pop(i)
print(f"Документов после очистки: {len(docs)}")

# Загрузка eval-данных
SKIP_PAGES = {72, 73, 74, 75, 76, 77, 78, 79, 80, 81}  # справочные таблицы

eval_data = EvalDataLoader.load(
    path="../context/docs_questions_qwen3_14b_awq.jsonl",
    question_types=["simple", "medium", "complex"],  # все типы вопросов
    skip_pages=SKIP_PAGES,
)


# ═════════════════════════════════════════════════════════════
# ЯЧЕЙКА 2: Создание раннера (один раз на сессию)
# ═════════════════════════════════════════════════════════════

runner = ExperimentRunner(
    docs=docs,
    eval_data=eval_data,
    embed_model_name="Qwen/Qwen3-Embedding-0.6B",
)


# ═════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Подбор chunk_size
# ═════════════════════════════════════════════════════════════

import numpy as np

chunk_sizes = np.linspace(150, 3000, 17, dtype=int).tolist()

configs_chunk_size = [
    RAGConfig(
        name=f"token_{cs}_200",
        chunking=ChunkingConfig(chunk_size=cs, chunk_overlap=200, splitter_type="token"),
    )
    for cs in chunk_sizes
]

results_cs = await runner.run_all(
    configs=configs_chunk_size,
    save_csv="../results/exp1_chunk_size.csv",
)

runner.plot(x_col="chunk_size", title="Эксперимент 1: Подбор chunk_size", save_path="../results/exp1_chunk_size.png")


# ═════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 2: Подбор overlap (при фиксированном chunk_size=500)
# ═════════════════════════════════════════════════════════════

overlaps = [50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400]

configs_overlap = [
    RAGConfig(
        name=f"token_500_{ov}",
        chunking=ChunkingConfig(chunk_size=500, chunk_overlap=ov, splitter_type="token"),
    )
    for ov in overlaps
]

results_ov = await runner.run_all(
    configs=configs_overlap,
    save_csv="../results/exp2_overlap.csv",
)

runner.plot(x_col="chunk_overlap", title="Эксперимент 2: Подбор overlap", save_path="../results/exp2_overlap.png")


# ═════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 3: Сравнение сплиттеров
# ═════════════════════════════════════════════════════════════

configs_splitters = [
    # TokenTextSplitter — baseline
    RAGConfig(
        name="token_500_200",
        chunking=ChunkingConfig(
            chunk_size=500, chunk_overlap=200, splitter_type="token",
        ),
    ),
    # SentenceSplitter — старается не рвать предложения
    RAGConfig(
        name="sentence_500_200",
        chunking=ChunkingConfig(
            chunk_size=500, chunk_overlap=200, splitter_type="sentence",
        ),
    ),
    # SemanticSplitter — режет по смысловым границам
    # chunk_size/overlap НЕ используются, размер определяется автоматически
    RAGConfig(
        name="semantic_90",
        chunking=ChunkingConfig(
            splitter_type="semantic", breakpoint_percentile=90,
        ),
    ),
    # SemanticSplitter с другим порогом
    RAGConfig(
        name="semantic_80",
        chunking=ChunkingConfig(
            splitter_type="semantic", breakpoint_percentile=80,
        ),
    ),
]

results_sp = await runner.run_all(
    configs=configs_splitters,
    save_csv="../results/exp3_splitters.csv",
)

# Для сравнения сплиттеров удобнее bar-plot или таблица,
# так как x-ось — категориальная
runner.plot(
    x_col="experiment",
    hue_col="splitter_type",
    title="Эксперимент 3: Сравнение сплиттеров",
    save_path="../results/exp3_splitters.png",
)


# # ═════════════════════════════════════════════════════════════
# # ЭКСПЕРИМЕНТ 4: Сравнение по типу вопроса (simple vs medium vs complex)
# # ═════════════════════════════════════════════════════════════

# # Для этого эксперимента загружаем данные отдельно по типам
# # и прогоняем лучший конфиг на каждом

# best_config = RAGConfig(
#     name="best_token_500_200",
#     chunking=ChunkingConfig(chunk_size=500, chunk_overlap=200, splitter_type="token"),
#     recreate=False,  # коллекция уже создана
# )

# for q_type in ["simple", "medium", "complex"]:
#     eval_subset = EvalDataLoader.load(
#         path="../context/docs_questions_qwen3_14b_awq.jsonl",
#         question_types=[q_type],
#         skip_pages=SKIP_PAGES,
#     )
    
#     runner_subset = ExperimentRunner(
#         docs=docs,
#         eval_data=eval_subset,
#         embed_model_name="Qwen/Qwen3-Embedding-0.6B",
#     )
    
#     results = await runner_subset.run_all(
#         configs=[RAGConfig(
#             name=f"best_{q_type}",
#             chunking=ChunkingConfig(chunk_size=500, chunk_overlap=200),
#             recreate=False,
#         )],
#         save_csv=f"../results/exp4_{q_type}.csv",
#     )
#     print(f"\n{q_type.upper()} вопросы:")
#     print(results[results["k"] == 5][["precision@k", "recall@k", "f1@k", "hit_rate@k", "mrr@k"]])


# # ═════════════════════════════════════════════════════════════
# # ДОПОЛНИТЕЛЬНО: Загрузка ранее сохранённых результатов
# # ═════════════════════════════════════════════════════════════

# import pandas as pd

# # Загрузить и визуализировать старые результаты
# df_old = pd.read_csv("../results/exp1_chunk_size.csv")
# ResultsVisualizer.plot_by_variable(
#     df=df_old,
#     x_col="chunk_size",
#     title="Эксперимент 1 (загружено из CSV)",
#     k_values=[1, 3, 5, 10],
# )
