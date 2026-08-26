import { Citation, Message, RAGDocument } from '../types';
import { INITIAL_CITATIONS, MOCK_DOCUMENTS } from './mockDocuments';

export interface RAGAnswer {
  content: string;
  citations: Citation[];
  retrievalLatencyMs: number;
  tokensCount: number;
  retrievedChunksCount: number;
}

export function generateRAGResponse(query: string, documents: RAGDocument[]): RAGAnswer {
  const normalized = query.toLowerCase();

  // Scenario 1: Sliding Window Attention / Mistral 7B Architecture
  if (normalized.includes('sliding') || normalized.includes('swa') || normalized.includes('memory') || normalized.includes('receptive') || normalized.includes('window')) {
    const citations: Citation[] = [
      INITIAL_CITATIONS[1],
      INITIAL_CITATIONS[2]
    ];
    return {
      content: `**Sliding Window Attention (SWA)** is one of the primary memory-saving mechanisms in **Mistral 7B** [1].

### Key Mechanics & Theoretical Receptive Field:
1. **Local Attention Window ($W = 4096$)**: Instead of full quadratic attention where each token attends to all previous tokens, each layer computes attention over a fixed window of $W = 4096$ tokens [1].
2. **Stacked Receptive Field**: Across $L = 32$ stacked transformer layers, the theoretical attention horizon reaches **up to $L \\times W \\approx 131,000$ tokens**, allowing long-range dependencies to propagate through the network while drastically bounding compute costs [1].
3. **KV Cache Compression**: SWA cuts the active Key-Value (KV) cache memory footprint by **more than 2.4x** [1].

Combined with **Grouped-Query Attention (GQA)** with 8 KV heads [2], Mistral 7B reduces 8k context KV cache size from 2.05 GB down to 0.82 GB, boosting generation throughput on standard 80GB A100 GPUs by **2.8x** [2].`,
      citations,
      retrievalLatencyMs: 185,
      tokensCount: 224,
      retrievedChunksCount: 4
    };
  }

  // Scenario 2: RAG Foundations / Parametric vs Non-Parametric (Lewis et al.)
  if (normalized.includes('parametric') || normalized.includes('non-parametric') || normalized.includes('foundation') || normalized.includes('lewis') || normalized.includes('rag')) {
    const citations: Citation[] = [
      INITIAL_CITATIONS[4],
      INITIAL_CITATIONS[5]
    ];
    return {
      content: `In standard Retrieval-Augmented Generation (RAG) models, memory is decoupled into two complementary systems [4]:

1. **Parametric Memory (The Generator)**:
   - Encoded directly inside the neural weights of a pre-trained sequence-to-sequence model (e.g. Mistral, BART, or T5).
   - Responsible for reasoning, linguistic fluency, code synthesis, and structured formatting [4].

2. **Non-Parametric Memory (The Vector Index)**:
   - A dense neural vector database over external knowledge chunks (e.g. Wikipedia articles, technical documentation, PDFs) [4].
   - Dense Passage Retrieval (DPR) maps questions and chunks into a shared 768-dimensional embedding space, executing Maximum Inner Product Search (MIPS) in sub-15ms [5].

### Core Advantage:
This architectural separation allows the non-parametric knowledge base to be **updated, redacted, or swapped in real-time** without requiring costly pre-training or fine-tuning runs [4].`,
      citations,
      retrievalLatencyMs: 142,
      tokensCount: 196,
      retrievedChunksCount: 5
    };
  }

  // Scenario 3: Grouped-Query Attention vs Multi-Head Attention
  if (normalized.includes('gqa') || normalized.includes('grouped') || normalized.includes('throughput') || normalized.includes('kv head') || normalized.includes('bandwidth')) {
    const citations: Citation[] = [
      INITIAL_CITATIONS[2],
      INITIAL_CITATIONS[3]
    ];
    return {
      content: `**Grouped-Query Attention (GQA)** addresses the severe memory-bandwidth bottleneck in autoregressive token generation [2].

- **Standard Multi-Head Attention (MHA)** allocates 1 Key and 1 Value head for every Query head (32:32 ratio), demanding large KV cache storage.
- **Mistral 7B GQA** groups 4 Query heads per single Key/Value head (a 4:1 ratio with 8 KV heads) [2].

### Performance & Latency Benefits:
- **Throughput Increase**: Delivers a **2.8x speedup** on 80GB A100 GPUs during multi-turn 8k context decoding [2].
- **Accuracy Parity**: Matches full MHA benchmarks on MMLU (60.1%) and GSM8k (52.2%) while drastically reducing GPU VRAM pressure [3].`,
      citations,
      retrievalLatencyMs: 168,
      tokensCount: 188,
      retrievedChunksCount: 3
    };
  }

  // Scenario 4: Hallucination mitigation / DPR
  if (normalized.includes('hallucinat') || normalized.includes('accuracy') || normalized.includes('benchmark') || normalized.includes('faith')) {
    const citations: Citation[] = [
      INITIAL_CITATIONS[6],
      INITIAL_CITATIONS[3]
    ];
    return {
      content: `Grounding language models with explicit dense vector passage retrieval directly mitigates factual hallucinations [6]:

- **68% Reduction in Hallucinations**: Human evaluation in the Lewis et al. benchmark demonstrated a **68% decrease in fabricated statements** compared to standalone parametric models [6].
- **State-of-the-Art Exact Match**: Achieves **44.5% Exact Match** on the Natural Questions benchmark [6].
- **Auditable Evidence**: Because every assertion links directly to an indexed chunk [6], readers can click verified references to inspect the exact passage in the source document [3].`,
      citations,
      retrievalLatencyMs: 210,
      tokensCount: 175,
      retrievedChunksCount: 4
    };
  }

  // Scenario 5: Mistral Large 2 / 128k context / Tool calling
  if (normalized.includes('large') || normalized.includes('128k') || normalized.includes('tool') || normalized.includes('function') || normalized.includes('needle')) {
    const citations: Citation[] = [
      INITIAL_CITATIONS[7],
      INITIAL_CITATIONS[8]
    ];
    return {
      content: `**Mistral Large 2** expands frontier capabilities with massive context processing and high-precision tool use [7]:

- **128k Token Context Window**: Maintains **99.4% retrieval accuracy** in Needle-In-A-Haystack evaluations across all token depths [7].
- **Parallel Function Calling**: Scores **84.0% on the Berkeley Function-Calling Benchmark (BFCL)**, excelling at strict JSON schema generation and multi-step agentic execution [8].
- **Multilingual Mastery**: Native support for 80+ programming and natural languages with competitive token economics [7].`,
      citations,
      retrievalLatencyMs: 195,
      tokensCount: 160,
      retrievedChunksCount: 3
    };
  }

  // Fallback dynamic response from matched documents
  const doc = documents[0] || MOCK_DOCUMENTS[0];
  const citation1: Citation = {
    id: 1,
    docId: doc.id,
    docTitle: doc.title,
    page: 1,
    snippet: doc.abstract || 'Document abstract and foundational methodology overview.',
    chunkTitle: 'Document Abstract & Scope',
    confidence: 0.91,
    score: 0.887,
    boundingBox: { top: 20, left: 6, width: 88, height: 18 }
  };

  const citation2: Citation = {
    id: 2,
    docId: doc.id,
    docTitle: doc.title,
    page: Math.min(2, doc.totalPages),
    snippet: doc.pages[1]?.content[1]?.text || 'Core technical architecture and empirical benchmarks.',
    chunkTitle: 'Core Methodology & Technical Evaluation',
    confidence: 0.89,
    score: 0.852,
    boundingBox: { top: 38, left: 6, width: 88, height: 18 }
  };

  return {
    content: `Based on an analysis of **${doc.title}** [1], here are the key findings relevant to your query *"${query}"*:

1. **System Foundation & Overview**: The indexed technical documentation details the architectural framework, tokenization protocols, and evaluation metrics [1].
2. **Empirical Results**: Retrieval-augmented synthesis confirms performance advantages and optimized inference throughput [2].

You can click the citation badges **[1]** and **[2]** to jump directly to the relevant passages in the document viewer.`,
    citations: [citation1, citation2],
    retrievalLatencyMs: 230,
    tokensCount: 145,
    retrievedChunksCount: 3
  };
}
