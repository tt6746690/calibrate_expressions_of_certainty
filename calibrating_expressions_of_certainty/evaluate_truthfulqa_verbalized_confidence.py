import os
import re
import json
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from .llm_api import get_completion_response
from .calibration import get_certainty_phrases


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--prompt_file', type=str, default=str(Path(__file__).parent / 'prompts/llm_calibration/verbconf.txt'))
    parser.add_argument('--first_n', type=int, default=0)
    parser.add_argument('--certainty_phrases_source', type=str, default='', help='only used if use prompt that verbalizes linguistic confidence.')
    args = parser.parse_args()

    dataset = load_dataset("truthful_qa", "generation", split="validation")
    if args.first_n > 0:
        dataset = dataset.select(range(args.first_n))

    prompt_name = os.path.basename(args.prompt_file).split('.txt')[0]

    if args.certainty_phrases_source:
        certainty_phrases = get_certainty_phrases(args.certainty_phrases_source)

    with open(args.prompt_file, 'r') as f:
        prompt_template = f.read()

    def create_prompt_fn(example):
        if prompt_name == 'verbprobtop1':
            prompt = prompt_template.format(
                question=example['question'],
            )
        else:
            prompt = prompt_template.format(
                expression_list=certainty_phrases,
                question=example['question'],
            )
        return {
            'prompt': prompt
        }
    dataset = dataset.map(create_prompt_fn)

    answers = []
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        prompt = dataset[i]['prompt']
        answer = get_completion_response(
            prompt,
            model=args.model,
            temperature=0,
            seed=0,
        )
        answers.append(answer)

    data = []
    for i, answer in enumerate(answers):
        example = dataset[i]
        
        if prompt_name == 'verbprobtop1':
            pattern = r"Guess:\s*(.+)\s*Probability:\s*(.+)\s*"
        else:
            pattern = r"Guess:\s*(.+)\s*Confidence:\s*(.+)\s*"
        match = re.search(pattern, answer)
        if match:
            guess = match.group(1).strip()
            confidence = match.group(2).strip()
        else:
            print(f"[{i}] No match found.")
            guess = ''
            confidence = ''

        data.append({
            'question': example['question'],
            'gold_answer': example['best_answer'],
            'correct_answers': example['correct_answers'],
            'pred_answer': guess,
            'confidence': confidence,
            'prompt': example['prompt'],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'answers.jsonl'), 'w') as f:
        for outputs in data:
            f.write(json.dumps(outputs) + '\n')
