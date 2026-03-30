import json
import datetime
from pathlib import Path



results_dir = Path.cwd() / "results"

context_dir = Path.cwd() / "context"

# функция сохранения результата в файл
def save_result(result, file_name: str) -> None:
    result_path = results_dir.joinpath(file_name)
    result["time"] = str(datetime.datetime.now())
    with open(result_path, "w") as f:
        f.write(json.dumps(result) + "\n")


def save_lists_to_json_file(title, questions:list, true_answers:list):
    path_to_save = context_dir.joinpath(title)
    with open(path_to_save, "a", encoding="utf-8") as f:
        for q, a in zip(questions, true_answers):
            obj = {"question":q, "true_answer":a}
            f.write(json.dumps(obj, ensure_ascii=False)+"\n")


def parse_json_to_lists(path):
    questions = []
    true_answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines:
            d = json.loads(line)
            questions.append(d["question"])
            true_answers.append(d["true_answer"])

    return questions, true_answers

questions, true_answers = parse_json_to_lists()