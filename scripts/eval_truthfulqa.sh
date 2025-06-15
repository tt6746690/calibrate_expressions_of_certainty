#!/bin/bash

set -e
set -x

model=$1
output_dir=$2
prompt_file=$3
first_n=$4
certainty_phrases_source=$5

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
prompts_dir="$(dirname "$script_dir")/calibrating_expressions_of_certainty/prompts/llm_calibration"

python -m calibrating_expressions_of_certainty.evaluate_truthfulqa_verbalized_confidence \
    --model $model \
    --output_dir $output_dir \
    --prompt_file $prompts_dir/$prompt_file.txt \
    --first_n $first_n \
    --certainty_phrases_source "$certainty_phrases_source"

python -m calibrating_expressions_of_certainty.evaluate_qa_correctness \
    --model gpt-4o-mini \
    --prompt_file $prompts_dir/check_answer_equivalence_multiple_correct.txt \
    --input_file $output_dir/answers.jsonl \
    --first_n $first_n