from datasets import load_dataset

print("Downloading FEVER dataset...")

dataset = load_dataset("fever", "v1.0")

print(dataset)

dataset.save_to_disk("data/fever")

print("Dataset saved successfully!")