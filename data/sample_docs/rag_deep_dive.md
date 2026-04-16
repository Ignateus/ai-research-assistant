# Retrieval-Augmented Generation — Deep Dive

## The Problem RAG Solves

LLMs are trained on static snapshots of data. Once trained, they have no awareness of:
- Events after their training cutoff
- Private or proprietary documents
- Real-time data (prices, news, sensor readings)

RAG bridges this gap by dynamically fetching relevant content at inference time and injecting it
into the model's context window.

## RAG Architecture

A production RAG system has two phases:

### Offline (Indexing) Phase
1. **Load** — ingest documents from files, databases, or APIs
2. **Chunk** — split documents into smaller segments (typically 256–1024 tokens)
3. **Embed** — convert each chunk into a vector using an embedding model
4. **Store** — persist vectors and metadata in a vector database (Chroma, Pinecone, Weaviate, etc.)

### Online (Query) Phase
1. **Embed query** — convert the user's question into a vector
2. **Retrieve** — find the top-k most similar chunks using approximate nearest-neighbour search
3. **Augment** — prepend retrieved chunks to the LLM prompt as context
4. **Generate** — the LLM produces an answer grounded in the retrieved content

## Chunking Strategies

Chunking strategy significantly affects retrieval quality:

- **Fixed-size chunking**: Split every N tokens with an overlap of M tokens. Simple and fast.
  Overlap avoids losing context at boundaries.
- **Sentence chunking**: Split on sentence boundaries. Better semantic coherence, variable size.
- **Recursive chunking**: Try paragraph splits first, fall back to sentence, then token.
- **Semantic chunking**: Use embedding similarity to find natural breakpoints.

## Embedding Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| all-MiniLM-L6-v2 | 384 | Fast, lightweight, good for general use |
| text-embedding-3-small | 1536 | OpenAI, strong performance |
| text-embedding-3-large | 3072 | OpenAI, best performance, higher cost |
| nomic-embed-text | 768 | Open source, competitive with commercial |

## Vector Databases

- **ChromaDB**: Open source, embeds locally, ideal for development and small deployments
- **Pinecone**: Managed cloud service, scales to billions of vectors
- **Weaviate**: Open source, supports hybrid search (vector + keyword)
- **pgvector**: Postgres extension — good if you already run Postgres

## Improving RAG Quality

### Query-side improvements
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer, embed it, use that
  vector for retrieval. Often retrieves better than embedding the raw question.
- **Query rewriting**: Ask the LLM to rephrase the query before retrieval.
- **Multi-query**: Generate multiple query variants, union the retrieved chunks.

### Retrieval improvements
- **Hybrid search**: Combine vector similarity with BM25 keyword search. Better for named entities.
- **Re-ranking**: Use a cross-encoder to re-score retrieved chunks before passing to the LLM.
- **Metadata filtering**: Pre-filter by date, author, or document type before vector search.

### Generation improvements
- **Citation**: Instruct the model to cite which source chunks it used.
- **Faithfulness checking**: Run a second LLM call to verify the answer is grounded in the context.

## Evaluation Metrics

| Metric | Measures |
|--------|---------|
| Retrieval precision | Fraction of retrieved chunks that are relevant |
| Retrieval recall | Fraction of relevant chunks that were retrieved |
| Answer faithfulness | Is the answer supported by the retrieved context? |
| Answer relevance | Does the answer address the user's question? |

Frameworks like RAGAS automate RAG evaluation using LLM-as-judge techniques.
