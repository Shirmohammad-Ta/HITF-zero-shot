

import json
import argparse
import random
from model import LLMWrapper
from selector import Selector
import numpy as np


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def construct_prompt(support_set, query):
    examples = "\n\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in support_set])
    return f"{examples}\n\nQ: {query['question']}\nA:"


def train(model, dataset, embeddings, output_path, k=8):
    selector = Selector(embeddings)

    predictions = []
    for query in dataset:
        task_emb = query['embedding']  # فرض: embedding قبلاً محاسبه شده و داخل query هست
        λ = selector.compute_lambda(task_emb)


        human_samples = [ex for ex in dataset if ex['source'] == 'human']
        self_samples = [ex for ex in dataset if ex['source'] == 'self']

        kh = int(np.ceil(λ * k))
        ks = k - kh

        support_h = random.sample(human_samples, min(kh, len(human_samples)))
        support_s = random.sample(self_samples, min(ks, len(self_samples)))
        support_set = support_h + support_s

        prompt = construct_prompt(support_set, query)
        response = model.query(prompt)

        predictions.append({
            "question": query['question'],
            "gold": query['answer'],
            "prediction": response
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/sample_superglue.json')
    parser.add_argument('--output', type=str, default='results/predictions.json')
    parser.add_argument('--model', type=str, default='gpt-3.5')
    parser.add_argument('--api', action='store_true')
    args = parser.parse_args()

    print(" Loading data...")
    dataset = load_dataset(args.data)

    print(" Loading model...")
    model = LLMWrapper(args.model, api=args.api)

    print(" Starting training/inference...")

    embeddings = {i: x['embedding'] for i, x in enumerate(dataset)}
    train(model, dataset, embeddings, args.output)
