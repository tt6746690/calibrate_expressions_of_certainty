import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

from .llm_api import get_completion_response


def load_precomputed_answers(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--prompt_file', type=str, default=str(Path(__file__).parent / 'prompts/llm_calibration/check_answer_equivalence.txt'))
    parser.add_argument('--first_n', type=int, default=0)
    args = parser.parse_args()

    if args.prompt_file.endswith('check_answer_equivalence.txt'):
        check_type = 'equivalence_with_best_answer'
    elif args.prompt_file.endswith('check_answer_equivalence_multiple_correct.txt'):
        check_type = 'equivalence_with_multiple_answers'
    else:
        raise ValueError(f'Invalid {args.prompt_file}')

    with open(args.prompt_file, 'r') as f:
        prompt_template = f.read()

    # Load precomputed answers
    precomputed_answers = load_precomputed_answers(args.input_file)
    if args.first_n > 0:
        precomputed_answers = precomputed_answers[:args.first_n]

    # Evaluate answers
    evaluated_data = []
    for entry in tqdm(precomputed_answers, desc="Evaluating answers"):

        if check_type == 'equivalence_with_best_answer':
            gold_answer = entry['gold_answer']
        elif check_type == 'equivalence_with_multiple_answers':
            gold_answer = '\n'.join([f'- {x}' for x in entry['correct_answers']])

        prompt = prompt_template.format(
            question=entry['question'],
            gold_answer=gold_answer,
            pred_answer=entry['pred_answer'],
        )
        
        response = get_completion_response(
            prompt,
            model=args.model,
            temperature=0,
            seed=0,
            max_tokens=400,
        )
        
        # Extract the Yes/No answer
        if check_type == 'equivalence_with_best_answer':
            answer = response.strip().split('.')[0].lower()
        elif check_type == 'equivalence_with_multiple_answers':
            answer = response.strip(' .').split('\n')[-1].lower()
        is_correct = 1 if "yes" in answer else 0
        
        evaluated_entry = entry.copy()
        evaluated_entry['is_correct'] = is_correct
        evaluated_entry['check_answer_equivalence_response'] = response
        evaluated_data.append(evaluated_entry)

    # Save results
    output_file = args.input_file.replace('.json', '_eval_correctness.json')
    
    with open(output_file, 'w') as f:
        for entry in evaluated_data:
            f.write(json.dumps(entry) + '\n')
