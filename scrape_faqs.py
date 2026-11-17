import json
import re
from scrapling.fetchers import Fetcher

# Target auto repair shop FAQ pages
urls = [
    "https://kwikkarlibertyhill.com/faq",
    "https://vicks-expertune.com/faq",
    "https://www.oilchangesancarlos.com/faq/",
    "https://www.lubecube.com/faq/",
]

qa_pairs = []

for url in urls:
    try:
        page = Fetcher.get(url, stealthy_headers=True)
        text = page.get_all_text() or page.markdown or ""
        print(f"=== {url} ===")
        print(f"Text length: {len(text)}")
        
        lines = text.split('\n')
        current_q = None
        current_a = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_question = False
            if line.endswith('?') and len(line) < 200:
                is_question = True
            elif line.startswith('###') and '?' in line:
                is_question = True
                line = line.replace('###', '').strip()
            elif line.startswith('Q:') or line.startswith('Q.'):
                is_question = True
                line = line.replace('Q:', '').replace('Q.', '').strip()
            
            if is_question:
                if current_q and current_a:
                    answer = ' '.join(current_a).strip()
                    if len(answer) > 10:
                        qa_pairs.append({
                            "question": current_q,
                            "answer": answer,
                            "source": url
                        })
                current_q = line
                current_a = []
            elif current_q:
                current_a.append(line)
        
        if current_q and current_a:
            answer = ' '.join(current_a).strip()
            if len(answer) > 10:
                qa_pairs.append({
                    "question": current_q,
                    "answer": answer,
                    "source": url
                })
                
    except Exception as e:
        print(f"Error scraping {url}: {e}")

print(f"\nTotal Q&A pairs extracted: {len(qa_pairs)}")

# Convert to PondLite JSONL format: instruction/input/output
records = []
for qa in qa_pairs:
    records.append({
        "instruction": qa["question"],
        "input": "",
        "output": qa["answer"],
        "source": qa["source"]
    })

# Write to file
output_path = "/Users/ojaihie/workspace/pondlite/raw/scraped_faq.jsonl"
with open(output_path, "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

print(f"\nWrote {len(records)} records to {output_path}")

# Print summary
for i, qa in enumerate(qa_pairs):
    print(f"\n{i+1}. Q: {qa['question'][:100]}")
    print(f"   A: {qa['answer'][:150]}...")
    print(f"   Source: {qa['source']}")
