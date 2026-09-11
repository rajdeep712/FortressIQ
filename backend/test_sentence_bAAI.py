from sentence_transformers import SentenceTransformer

# 1. Use the official Hugging Face string identifier.
# SentenceTransformers checks your local .cache folder first and loads it automatically.
model_path = "BAAI/bge-base-en-v1.5"

print("Loading BGE model from local cache...")
model = SentenceTransformer(model_path)

# 2. Match the model's native maximum capacity.
# 512 tokens comfortably accommodates your ~2000 character sentences.
model.max_seq_length = 512 

# Mock data: 30 sentences (~2000 characters each)
dummy_sentences = [
    "This is a long sentence sample for BGE. " * 60, 
    "Another distinct batch sentence for evaluation. " * 60,
    "More data for testing BGE embedding logic. " * 60
] * 10  

def get_native_embeddings(sentences, batch_size=32):
    print(f"Processing {len(sentences)} sentences in batches of {batch_size}...")
    
    # Generate native 768-dimensional embeddings
    embeddings = model.encode(
        sentences, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    return embeddings

# Execute batch embedding generation
final_embeddings = get_native_embeddings(dummy_sentences, batch_size=8)

print(f"\nSuccess! Generated array shape: {final_embeddings.shape}")
# Expected Output Shape: (30, 768)
