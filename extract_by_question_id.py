import json

# Load medqa_adaptive_331samples.json and extract sample numbers
adaptive_path = 'output/medqa_adaptive_331samples.json'
adaptive_samples = []
with open(adaptive_path, 'r', encoding='utf-8') as f:
    adaptive_samples = json.load(f)

# Extract sample numbers from question_id (e.g., 'sample_3' -> 3)
sample_numbers = set()
for sample in adaptive_samples:
    qid = sample.get('question_id', '')
    if qid.startswith('sample_'):
        try:
            num = int(qid.split('_')[1])
            sample_numbers.add(num)
        except Exception:
            continue

# Read all lines from test.jsonl
input_path = 'data/medqa/test.jsonl'
output_path = 'data/medqa/test_adaptive_331samples.jsonl'
with open(input_path, 'r', encoding='utf-8') as fin:
    test_lines = fin.readlines()

# Write only the lines whose (1-based) index is in sample_numbers
count = 0
with open(output_path, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(test_lines, 1):
        if idx in sample_numbers:
            fout.write(line)
            count += 1
print(f"Matched and wrote {count} samples.") 