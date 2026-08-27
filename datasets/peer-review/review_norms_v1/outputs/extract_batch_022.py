import json
import re

# Load input
with open('/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/review_norms_v1/inputs/batch_022.json') as f:
    reviews = json.load(f)

def extract_passages(review_text):
    """Extract all evaluative passages from a review."""
    passages = []

    # Find Strengths and Weaknesses sections
    strength_match = re.search(r'\*\*Strengths?:\*\*(.*?)(?=\*\*Weaknesses?:|\*\*Weakness:|\Z)',
                               review_text, re.DOTALL | re.IGNORECASE)
    weakness_match = re.search(r'\*\*Weaknesses?:|\*\*Weakness:\*\*(.*?)(?=\*\*[A-Z]|\Z)',
                               review_text, re.DOTALL | re.IGNORECASE)

    # Extract from strengths
    if strength_match:
        strengths_text = strength_match.group(1).strip()
        passages.extend(extract_from_section(strengths_text, 'pos', review_text))

    # Extract from weaknesses
    if weakness_match:
        weaknesses_text = weakness_match.group(1).strip()
        passages.extend(extract_from_section(weaknesses_text, 'neg', review_text))

    # Also extract from main body (before sections or if no sections)
    if not strength_match and not weakness_match:
        passages.extend(extract_from_main_body(review_text))

    return passages

def extract_from_section(section_text, polarity, full_review_text):
    """Extract passages from a structured section."""
    passages = []

    # Split by bullet points or numbered items
    # Pattern: start of line with -, +, *, number, or just paragraphs
    items = re.split(r'\n(?:[-+*]\s*|\d+[.)]\s*|`[+-]`\s*)', section_text)

    # If no bullet structure, split by double newlines
    if len(items) <= 1:
        items = section_text.split('\n\n')

    for item in items:
        item = item.strip()
        if len(item) < 15:
            continue

        # Split into sentences for finer granularity
        sentences = split_sentences(item)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            # Check if this is truly evaluative
            if is_evaluative(sent):
                # Verify it's an exact substring
                if sent in full_review_text:
                    quote = sent
                    if len(quote) > 300:
                        # Try to find a good truncation point
                        quote = truncate_to_evaluative_core(quote)

                    aspect = extract_aspect_from_text(sent)

                    passages.append({
                        'quote': quote,
                        'polarity': polarity,
                        'aspect': aspect
                    })

    return passages

def extract_from_main_body(review_text):
    """Extract evaluative passages from unstructured review."""
    passages = []

    # Split into sentences
    sentences = split_sentences(review_text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue

        polarity = detect_polarity(sent)
        if polarity and is_evaluative(sent):
            quote = sent
            if len(quote) > 300:
                quote = truncate_to_evaluative_core(quote)

            aspect = extract_aspect_from_text(sent)

            passages.append({
                'quote': quote,
                'polarity': polarity,
                'aspect': aspect
            })

    return passages

def split_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return sentences

def is_evaluative(text):
    """Check if text contains evaluative language."""
    text_lower = text.lower()

    # Evaluative patterns
    eval_patterns = [
        r'\b(good|bad|strong|weak|excellent|poor|effective|ineffective|novel|significant)\b',
        r'\b(clear|unclear|confusing|well-written|difficult)\b',
        r'\b(thorough|comprehensive|detailed|insufficient|lacking)\b',
        r'\b(original|innovative|incremental|straightforward)\b',
        r'\b(sound|rigorous|correct|incorrect|valid|invalid)\b',
        r'\b(important|meaningful|relevant|practical)\b',
        r'\b(better|worse|superior|inferior|outperform)\b',
    ]

    return any(re.search(pattern, text_lower) for pattern in eval_patterns)

def detect_polarity(text):
    """Detect polarity of evaluation."""
    text_lower = text.lower()

    pos_words = ['good', 'strong', 'excellent', 'effective', 'novel', 'significant',
                 'thorough', 'comprehensive', 'clear', 'well', 'better', 'superior',
                 'outperform', 'sound', 'rigorous', 'original', 'important']

    neg_words = ['weak', 'poor', 'ineffective', 'limited', 'lack', 'insufficient',
                 'unclear', 'confusing', 'difficult', 'worse', 'inferior', 'missing']

    pos_count = sum(1 for word in pos_words if word in text_lower)
    neg_count = sum(1 for word in neg_words if word in text_lower)

    if pos_count > neg_count:
        return 'pos'
    elif neg_count > pos_count:
        return 'neg'
    elif pos_count > 0 and neg_count > 0:
        return 'mixed'

    return None

def truncate_to_evaluative_core(text):
    """Truncate long text to evaluative core, max 300 chars."""
    # Find the first evaluative word and include context around it
    if len(text) <= 300:
        return text

    # Take first 297 chars and add ellipsis
    return text[:297] + '...'

def extract_aspect_from_text(text):
    """Extract the criterion/aspect being evaluated."""
    text_lower = text.lower()

    # Map patterns to aspect names (using reviewer's framing)
    aspect_map = {
        'novelty': ['novel', 'new', 'original', 'innovation', 'first attempt', 'unprecedented'],
        'writing clarity': ['clear', 'easy to follow', 'well-written', 'well written', 'understandable'],
        'experimental evaluation': ['experiment', 'evaluation', 'testing', 'validation', 'empirical', 'result'],
        'technical soundness': ['sound', 'correct', 'rigorous', 'proof', 'technical'],
        'significance': ['important', 'significant', 'meaningful', 'impact'],
        'contribution': ['contribution', 'advance'],
        'related work': ['related work', 'literature review', 'prior work'],
        'methodology': ['method', 'approach', 'technique', 'algorithm', 'framework', 'design'],
        'reproducibility': ['reproduce', 'code', 'implementation detail'],
        'computational efficiency': ['efficient', 'computational', 'runtime', 'speed', 'complexity', 'time'],
        'thoroughness': ['comprehensive', 'thorough', 'extensive', 'detailed'],
        'motivation': ['motivation', 'motivat', 'problem'],
        'generalizability': ['generaliz', 'broader', 'applicability', 'scope'],
        'clarity of presentation': ['presentation', 'organized', 'structured'],
        'incremental contribution': ['incremental', 'straightforward combination', 'limited novelty'],
        'insufficient analysis': ['insufficient', 'lacking', 'lack detail', 'missing'],
        'completeness': ['complete', 'missing', 'comprehensive'],
        'scalability': ['scale', 'scalab', 'large'],
    }

    # Find best matching aspect
    for aspect, keywords in aspect_map.items():
        if any(kw in text_lower for kw in keywords):
            return aspect

    # Default fallback
    return 'overall quality'

# Process all reviews
results = []
total_passages = 0

for review in reviews:
    passages = extract_passages(review['review_text'])
    total_passages += len(passages)

    results.append({
        'review_id': review['review_id'],
        'paper_id': review['paper_id'],
        'passages': passages
    })

# Write output
with open('/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/review_norms_v1/outputs/batch_022.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

# Print stats
total_reviews = len(results)
avg_passages = total_passages / total_reviews if total_reviews > 0 else 0

print(f"{total_reviews} reviews, {total_passages} passages ({avg_passages:.1f} per review)")
