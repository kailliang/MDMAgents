import json
import re
import csv

input_filename = 'output/flash_lite_med_bas_10_smpl.json'
output_filename = 'evaluation/flash_lite_med_bas_10_smpl.csv'

with open(input_filename, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Handle new JSON structure with metadata and results
if isinstance(raw_data, dict) and 'results' in raw_data:
    data = raw_data['results']
elif isinstance(raw_data, list):
    data = raw_data
else:
    raise ValueError("Unexpected JSON structure")
def extract_final_answer_or_answer(text):
    # Handle parse error cases first
    if 'Parse error' in text:
        parse_error_match = re.search(r'([A-EX])\)\s*Parse error', text)
        if parse_error_match:
            return 'X'  # Mark parse errors as 'X' for tracking
    
    # Check for "Answer: X" pattern anywhere in the text (common in majority_vote responses)
    answer_pattern = re.search(r'Answer:\s*([A-E])(?:\b|$)', text, re.IGNORECASE)
    if answer_pattern:
        return answer_pattern.group(1)
    
    match_final = re.search(r'(The final answer.*?[\.\n])', text, re.IGNORECASE)
    if match_final:
        text = match_final.group(1).strip()
    else:
        match_answer = re.search(r'(\n\n\*\*Answer:\*\* *\([A-E]\).*)', text)
        if match_answer:
            text = match_answer.group(1).strip()
        else:
            words = text.strip().split()
            # For short texts, use full text. For long texts, prefer first 15 words where answers usually appear
            if len(words) <= 15:
                last15 = text.strip()
            else:
                # Try first 15 words first (where answers usually are), then last 15 as fallback
                first15 = ' '.join(words[:15])
                last15 = first15
            match_correct = re.search(r'(The correct answer is \*\*\([A-E]\)\*\*)', last15)
            if match_correct:
                text = match_correct.group(1).strip()
            else:
                match_star = re.search(r'(\*\*\([A-E]\)\*\*)', last15)
                if match_star:
                    text = match_star.group(1).strip()
                else:
                    match_diag = re.search(r'(\*\*The most likely diagnosis is \([A-E]\)\*\*)', last15)
                    if match_diag:
                        text = match_diag.group(1).strip()
                    else:
                        text = last15
    
    # Enhanced letter extraction with multiple patterns
    # Pattern 1: Standard format with word boundaries
    match_letter = re.search(r'\b([A-E])\b', text)
    if match_letter:
        return match_letter.group(1)
    
    # Pattern 2: Parentheses format like (A) 
    match_paren = re.search(r'\(([A-E])\)', text)
    if match_paren:
        return match_paren.group(1)
    
    # Pattern 3: Letter followed by closing parenthesis like A)
    match_close_paren = re.search(r'([A-E])\)', text)
    if match_close_paren:
        return match_close_paren.group(1)
    
    # Pattern 4: Any single letter A-E (last resort)
    match_any = re.search(r'([A-E])', text)
    if match_any:
        return match_any.group(1)
    
    return ''

processed_data = []
total_correct = 0
total_samples = len(data)

# Track correct/total for each difficulty
correct_by_diff = {'basic': 0, 'intermediate': 0, 'advanced': 0}
total_by_diff = {'basic': 0, 'intermediate': 0, 'advanced': 0}

for sample in data:
    question_id = sample.get('question_id', '')
    label = sample.get('ground_truth', sample.get('label', ''))
    resp = sample.get('model_response', sample.get('response', ''))

    response_text = ''
    if isinstance(resp, dict):
        if 'final_answer' in resp:
            response_text = resp['final_answer']
        else:
            response_text = resp.get('0.0', '')
            if not response_text and 'majority_vote' in resp:
                response_text = resp['majority_vote']
    else:
        response_text = str(resp)
    
    response = extract_final_answer_or_answer(response_text)

    difficulty = sample.get('difficulty', '')
    token_usage = sample.get('token_usage', {})
    input_tokens = token_usage.get('input_tokens', '')
    output_tokens = token_usage.get('output_tokens', '')
    total_tokens = token_usage.get('total_tokens', '')
    
    is_correct = 1 if label == response else 0
    total_correct += is_correct

    # Track per-difficulty
    if difficulty in correct_by_diff:
        correct_by_diff[difficulty] += is_correct
        total_by_diff[difficulty] += 1

    processed_data.append([
        question_id, label, response, difficulty, 
        input_tokens, output_tokens, total_tokens, is_correct
    ])

accuracy = total_correct / total_samples if total_samples > 0 else 0

# Report per-difficulty accuracy
for diff in ['basic', 'intermediate', 'advanced']:
    total = total_by_diff[diff]
    correct = correct_by_diff[diff]
    acc = correct / total if total > 0 else 0
    print(f'Accuracy for {diff}: {acc:.4f} ({correct}/{total})')

with open(output_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ["question_id", "label", "response", "difficulty", "input_tokens", "output_tokens", "total_tokens", "correct"]
    writer.writerow(header)
    writer.writerows(processed_data)

print(f'Accuracy: {accuracy:.4f}')
print(f'CSV exported: {output_filename}')
