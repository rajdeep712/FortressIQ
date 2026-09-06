from fastembed import SparseTextEmbedding
child_chunks = [
    "This is the first 2000 character child chunk...",
    "This is the second 2000 character child chunk...",
    "This is the third 2000 character child chunk...",
]



# Load BM25 sparse encoder locally
bm25_model = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)


# Generate sparse vectors
embeddings = list(
    bm25_model.embed(child_chunks)
)


for i, embedding in enumerate(embeddings):

    print(f"\nChild chunk {i}")

    print("Indices:")
    print(embedding.indices)

    print("Values:")
    print(embedding.values)