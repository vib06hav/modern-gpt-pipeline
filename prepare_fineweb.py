import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    # We don't need billions of tokens for our ~30M parameter model test.
    # 50,000 documents is roughly 50-100MB of high quality text.
    num_docs = 50000
    
    print("Downloading FineWeb-Edu subset...")
    # Streaming allows us to just grab the first N documents without downloading the entire 100GB dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    os.makedirs("data", exist_ok=True)
    out_file = "data/fineweb_subset.txt"
    
    print(f"Writing {num_docs} documents to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(tqdm(dataset, total=num_docs)):
            if i >= num_docs:
                break
            # Write the text and separate documents with an end-of-text marker
            f.write(doc["text"] + "\n<|endoftext|>\n")
            
    print(f"Done! Dataset is ready at {out_file}.")

if __name__ == "__main__":
    main()
