import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from rag_pipeline import generate_answer

TEST_CASES = [
    {
        "question": "What is the purpose of this document?",
        "answer": "Answer based on your actual PDF documents.",
        "contexts": ["Replace this example with a real retrieved context."],
    },
]


def run_evaluation():
    if not TEST_CASES:
        raise RuntimeError("Add 15–20 question/answer examples to TEST_CASES.")

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for case in TEST_CASES:
        result = generate_answer(case["question"])
        rows["question"].append(case["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([source["snippet"] for source in result["sources"]])
        rows["ground_truth"].append(case["answer"])

    dataset = Dataset.from_dict(rows)
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    print(scores)


if __name__ == "__main__":
    run_evaluation()
