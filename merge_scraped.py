import json
import re

# Read scraped data
with open("/Users/ojaihie/workspace/pondlite/raw/scraped_faq.jsonl") as f:
    scraped = [json.loads(line) for line in f]

# Read existing data
with open("/Users/ojaihie/workspace/pondlite/raw/autorepair_shop.jsonl") as f:
    existing = [json.loads(line) for line in f]

# Noise patterns to filter out
noise_questions = {
    "?", "still have questions?", "ready for the fastest oil change in san carlos?",
}

noise_patterns = [
    r"get directions",
    r"call \d",
    r"book service",
    r"contact info",
]

cleaned = []
seen_questions = set()

# Add existing questions to seen set
for rec in existing:
    seen_questions.add(rec["instruction"].lower().strip())

for qa in scraped:
    q = qa["instruction"].strip().lower()
    a = qa["output"].strip()

    # Skip noise
    if q in noise_questions:
        continue
    if len(q) < 5:
        continue
    if any(p in q for p in noise_patterns):
        continue
    if q in seen_questions:
        continue  # Already covered by existing Fairview data

    # Generalize competitor-specific details
    # Replace phone numbers with generic placeholder
    a = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', a)
    # Replace specific addresses
    a = re.sub(r'\d+ [\w\s]+ (?:Street|St|Avenue|Ave|Road|Rd|Highway|Hwy|Lane|Ln|Drive|Dr|Boulevard|Blvd|Parkway|Pkwy)[\w\s,]*', '[ADDRESS]', a)
    # Replace specific city names with generic
    a = re.sub(r'(Liberty Hill|Lincoln Park|San Carlos|Norwood|Austin)', '[CITY]', a)
    # Replace state abbreviations
    a = re.sub(r'\b(TX|MI|CA|MA|OH)\b', '[STATE]', a)
    # Replace specific business names
    a = re.sub(r'(Kwik Kar|Vick.?s Expertune|Perfect Lube|Lube Cube|No Name Auto Repair)', '[SHOP]', a)

    cleaned.append({
        "instruction": qa["instruction"].strip(),
        "input": "",
        "output": a,
        "source": "scraped_competitor_faq"
    })
    seen_questions.add(q)

print(f"Existing records: {len(existing)}")
print(f"Scraped records: {len(scraped)}")
print(f"Cleaned new records: {len(cleaned)}")
print(f"Total after merge: {len(existing) + len(cleaned)}")

# Show what we're adding
for i, rec in enumerate(cleaned):
    print(f"\n{i+1}. Q: {rec['instruction'][:90]}")
    print(f"   A: {rec['output'][:130]}...")

# Merge and write
merged = existing + cleaned
output_path = "/Users/ojaihie/workspace/pondlite/raw/autorepair_shop.jsonl"
with open(output_path, "w") as f:
    for rec in merged:
        f.write(json.dumps(rec) + "\n")

print(f"\nWrote {len(merged)} total records to {output_path}")
