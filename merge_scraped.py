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

# Negative examples: out-of-domain questions that should trigger abstention
# These teach the TF-IDF retriever to say "we don't do that" instead of
# wrongly matching an auto repair answer for non-automotive questions.
negative_examples = [
    {"instruction": "Do you sell groceries?", "input": "", "output": "We don't sell groceries — we're an auto repair shop. We handle oil changes, brake repair, tires, and other automotive services. Call us at [PHONE] for car service."},
    {"instruction": "Do you sell clothing?", "input": "", "output": "We don't sell clothing — we're an auto repair shop. We service and repair vehicles, not apparel. Call us at [PHONE]."},
    {"instruction": "Do you deliver food?", "input": "", "output": "We don't deliver food — we're an auto repair shop. We fix cars, not meals. Call us at [PHONE] for automotive service."},
    {"instruction": "Can I book a hotel room?", "input": "", "output": "We don't book hotel rooms — we're an auto repair shop. We repair and maintain vehicles. Call us at [PHONE]."},
    {"instruction": "Do you sell electronics?", "input": "", "output": "We don't sell electronics — we're an auto repair shop. We service cars and trucks. Call us at [PHONE]."},
    {"instruction": "Do you sell furniture?", "input": "", "output": "We don't sell furniture — we're an auto repair shop. We work on vehicles, not home goods. Call us at [PHONE]."},
    {"instruction": "Can I buy a plane ticket?", "input": "", "output": "We don't sell plane tickets — we're an auto repair shop. We fix cars, not travel. Call us at [PHONE]."},
    {"instruction": "Do you sell pets?", "input": "", "output": "We don't sell pets — we're an auto repair shop. We service vehicles. Call us at [PHONE] for car care."},
    {"instruction": "Can I order pizza?", "input": "", "output": "We don't sell food or deliver pizza — we're an auto repair shop. We repair cars and trucks. Call us at [PHONE]."},
    {"instruction": "Do you sell insurance?", "input": "", "output": "We don't sell insurance — we're an auto repair shop. We work on vehicles, not policies. Call us at [PHONE]."},
    {"instruction": "Do you sell shoes?", "input": "", "output": "We don't sell shoes — we're an auto repair shop. We service and repair vehicles, not footwear. Call us at [PHONE]."},
    {"instruction": "Do you sell books?", "input": "", "output": "We don't sell books — we're an auto repair shop. We work on cars and trucks, not books. Call us at [PHONE]."},
    {"instruction": "Do you sell toys?", "input": "", "output": "We don't sell toys — we're an auto repair shop. We fix vehicles, not sell toys. Call us at [PHONE]."},
    {"instruction": "Do you sell jewelry?", "input": "", "output": "We don't sell jewelry — we're an auto repair shop. We service vehicles, not sell accessories. Call us at [PHONE]."},
    {"instruction": "Do you sell hardware?", "input": "", "output": "We don't sell hardware — we're an auto repair shop. We work on vehicles, not home improvement supplies. Call us at [PHONE]."},
    {"instruction": "Are you a grocery store?", "input": "", "output": "No, we're not a grocery store — we're an auto repair shop. We repair and maintain vehicles. Call us at [PHONE]."},
    {"instruction": "Are you a restaurant?", "input": "", "output": "No, we're not a restaurant — we're an auto repair shop. We fix cars, not serve food. Call us at [PHONE]."},
    {"instruction": "Do you sell anything besides auto parts?", "input": "", "output": "No, we only sell automotive parts and services — we're an auto repair shop. Call us at [PHONE]."},
    {"instruction": "Do you sell home appliances?", "input": "", "output": "We don't sell home appliances — we're an auto repair shop. We service vehicles, not refrigerators or washers. Call us at [PHONE]."},
    {"instruction": "Do you sell garden supplies?", "input": "", "output": "We don't sell garden supplies — we're an auto repair shop. We work on cars and trucks. Call us at [PHONE]."},
    {"instruction": "Can I walk in without an appointment?", "input": "", "output": "Yes, walk-ins are welcome. We recommend calling ahead at [PHONE] so we can give you a realistic wait time."},
    {"instruction": "Do you accept walk-in customers?", "input": "", "output": "Yes, walk-ins are welcome. We recommend calling ahead at [PHONE] so we can give you a realistic wait time."},
]

# Add negative examples to existing data (they're part of the base training set)
for rec in negative_examples:
    existing.append(rec)
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
