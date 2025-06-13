

import argparse
from model import LLMWrapper
import json
import os


def load_prompts(path):
    with open(path, 'r', encoding='utf-8') as f:
        blocks = f.read().strip().split("\n\n")
        prompts = []
        for block in blocks:
            lines = block.strip().split("\n")
            q_line = next((l for l in lines if l.startswith("Q:")), "")
            a_line = next((l for l in lines if l.startswith("A:")), "")
            question = q_line[2:].strip()
            answer = a_line[2:].strip()
            prompts.append((question, answer))
        return prompts


def evaluate(model, prompts, output_path):
    results = []
    for i, (question, gold) in enumerate(prompts):
        full_prompt = f"Q: {question}\nA:"
        prediction = model.query(full_prompt)
        results.append({
            "id": i,
            "question": question,
            "gold": gold,
            "prediction": prediction
        })
        print(f"[{i}]  Done")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, default='prompts/example_prompts.txt')
    parser.add_argument('--output', type=str, default='results/eval_results.json')
    parser.add_argument('--model', type=str, default='gpt-3.5')
    parser.add_argument('--api', action='store_true')
    args = parser.parse_args()

    print(" Loading prompts...")
    prompts = load_prompts(args.prompt_file)

    print(" Loading model...")
    model = LLMWrapper(args.model, api=args.api)

    print(" Starting evaluation...")
    evaluate(model, prompts, args.output)
