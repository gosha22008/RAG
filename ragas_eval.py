import pandas as pd
from ragas.run_config import RunConfig
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from pathlib import Path
from llama_index.core import query_engine, Settings
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from functions import save_result

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# LLM_MODEL_NAME = "Qwen/Qwen3-32B-AWQ"

# true_file_path = Path.cwd() / "data1"

# file = true_file_path.joinpath("QuestionsWithTrueAnswers.txt")


# # читаем файл
# with open(file, "r", encoding="utf-8") as f:
#     questions_true_answers = f.readlines()

# questions = []
# true_answers = []
# for qa in questions_true_answers:
#     qa = qa.strip()
#     q_index = qa.find("?") + 1
#     questions.append(qa[:q_index].strip().split(".")[1].strip())
#     true_answers.append(qa[q_index:].strip())




# results = []

# for q in questions:
#     response = query_engine.query(q) 
    
#     results.append({
#         "question": q,
#         "answer": response.response,
#         "contexts": [node.node.get_content() for node in response.source_nodes],
#         "ground_truth": true_answers[questions.index(q)]
#     })

# df_eval = pd.DataFrame(results)

df_eval = pd.read_csv("results/df_eval", index_col=0)


run_config = RunConfig(
    timeout=60, 
    # max_workers=2, # ограничиваем одним потоком (ограничения из-за локального запуска)
    max_retries=10
)


# подключение embedding модели с CPU
Settings.embed_model = TextEmbeddingsInference(model_name=EMBED_MODEL_NAME,
                                               base_url="http://localhost:8081")




ragas_llm = LlamaIndexLLMWrapper(llm=Settings.llm)
ragas_embeds = LlamaIndexEmbeddingsWrapper(embeddings=Settings.embed_model)

dataset = Dataset.from_pandas(df_eval)

score = evaluate(
    dataset,
    llm=ragas_llm, 
    embeddings=ragas_embeds,
    run_config=run_config 
)


result = score.to_pandas()[["answer_relevancy", "context_precision", "faithfulness", "context_recall"]].agg("mean", axis=0)

result = result.to_dict()
# result["latency"] = latency

save_result(result, "baseline_vllm_fp8")