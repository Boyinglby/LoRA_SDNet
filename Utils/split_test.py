import json, random

# Load CoQA dev set
with open("../coqa/data/coqa-dev-v1.0_orig.json", "r") as f:
    dev = json.load(f)

# Shuffle for randomness
random.seed(42)
random.shuffle(dev["data"])

# Split 80% validation, 20% test
split_idx = int(0.5 * len(dev["data"]))
dev_small = {"data": dev["data"][:split_idx], "version": dev["version"]}
test_small = {"data": dev["data"][split_idx:], "version": dev["version"]}

# Save new files
with open("../coqa/data/coqa-dev-v1.0.json", "w") as f:
    json.dump(dev_small, f)
with open("../coqa/data/coqa-test.json", "w") as f:
    json.dump(test_small, f)

print(f"Validation size: {len(dev_small['data'])}, Test size: {len(test_small['data'])}")
