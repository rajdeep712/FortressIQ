import { RAGDocument, PresetPrompt, Citation } from '../types';

export const MOCK_DOCUMENTS: RAGDocument[] = [
  {
    id: 'mistral-7b-v01',
    title: 'Mistral 7B: Efficient Open Weight Foundation Model',
    filename: 'mistral-7b-v0.1.pdf',
    totalPages: 5,
    fileType: 'pdf',
    uploadedAt: 'Today at 09:30 AM',
    fileSize: '2.4 MB',
    category: 'Architecture & Optimization',
    authors: 'Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, et al.',
    abstract: 'We introduce Mistral 7B, a 7-billion-parameter language model engineered for superior performance and high inference efficiency. Mistral 7B leverages Grouped-query attention (GQA) for faster inference and Sliding Window Attention (SWA) to handle arbitrarily long sequences with reduced memory footprint. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks.',
    pages: [
      {
        pageNumber: 1,
        title: '1. Introduction & Executive Abstract',
        sectionHeader: 'MISTRAL AI TECHNICAL REPORT • SEPTEMBER 2023',
        content: [
          {
            type: 'heading',
            text: 'Mistral 7B: Architecture and Empirical Evaluation'
          },
          {
            type: 'callout',
            text: 'Abstract — We present Mistral 7B, a 7.3B parameter model that outperforms Llama 2 13B on all benchmarks and matches Llama 1 34B on many metrics. It features Sliding Window Attention (SWA) to reduce memory costs during generation while maintaining 8k context window capability.'
          },
          {
            type: 'paragraph',
            text: 'Recent progress in Large Language Models (LLMs) has primarily focused on scaling parameters. However, operational efficiency at inference time is critical for real-world deployment. In this work, we demonstrate that a thoughtfully designed architecture with high computational density can surpass models with twice its parameter count.'
          },
          {
            type: 'subheading',
            text: '1.1 Key Architectural Innovations'
          },
          {
            type: 'bullet-list',
            items: [
              'Sliding Window Attention (SWA): Attention is computed over a sliding window of size W = 4096, reducing KV cache footprint by 2.4x.',
              'Grouped-Query Attention (GQA): 8 key-value query heads reduce memory bandwidth constraints during decoding.',
              'Byte-fallback BPE Tokenizer: 32k vocabulary with native UTF-8 byte level fallback for robust multilingual handling.'
            ]
          }
        ]
      },
      {
        pageNumber: 2,
        title: '2. Sliding Window Attention & KV Cache Reduction',
        sectionHeader: 'SECTION 2 • MEMORY & ATTENTION MECHANISMS',
        content: [
          {
            type: 'heading',
            text: '2. Sliding Window Attention (SWA)'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 1,
            chunkMeta: {
              id: 'chunk-swa-mem',
              confidence: 0.96,
              label: 'SWA Memory Reduction & Layer Receptive Field'
            },
            text: 'Sliding Window Attention (SWA) limits the attention span of each token to a local window of size W. In a network with L layers, the theoretical receptive field of token k at layer L reaches up to L × W tokens. For W = 4096 and L = 32, this gives a theoretical attention horizon of ~131,000 tokens through stacked multilayer representations, while cutting the memory footprint of the Key-Value (KV) cache down by more than 2.4x compared to full attention.'
          },
          {
            type: 'equation',
            text: 'Receptive Field(l) = min(l × W, Sequence Length)  where W = 4096, l ∈ [1, 32]'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 2,
            chunkMeta: {
              id: 'chunk-gqa-speed',
              confidence: 0.92,
              label: 'Grouped-Query Attention Speedups'
            },
            text: 'Alongside SWA, Mistral 7B incorporates Grouped-Query Attention (GQA) with 8 KV heads for 32 query heads (a 4:1 query-to-key ratio). This drastically reduces the size of the key-value cache during generation, allowing higher batch sizes and yielding a 2.8x throughput increase on standard 80GB A100 GPUs during long-sequence generation.'
          },
          {
            type: 'table',
            caption: 'Table 1: Memory footprint and decode latency comparison for 8k context generation.',
            items: [
              'Model | KV Cache Size (8k) | Throughput (tok/s) | GPU Memory (VRAM)',
              'Llama-2 7B (Full MHA) | 2.05 GB | 38.4 | 14.8 GB',
              'Mistral 7B (GQA + SWA) | 0.82 GB | 108.2 | 8.1 GB (-45%)'
            ]
          }
        ]
      },
      {
        pageNumber: 3,
        title: '3. Empirical Benchmarks & Comparative Evaluation',
        sectionHeader: 'SECTION 3 • REASONING, CODE & MATH BENCHMARKS',
        content: [
          {
            type: 'heading',
            text: '3. Comprehensive Benchmark Results'
          },
          {
            type: 'paragraph',
            text: 'We evaluate Mistral 7B against Llama 2 7B, Llama 2 13B, and CodeLlama 7B across standard academic benchmarks spanning common sense reasoning, world knowledge, reading comprehension, mathematics, and code generation.'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 3,
            chunkMeta: {
              id: 'chunk-benchmarks',
              confidence: 0.94,
              label: 'MMLU & GSM8k Performance vs Llama 2'
            },
            text: 'On MMLU (Massive Multitask Language Understanding), Mistral 7B achieves 60.1% accuracy, outperforming Llama 2 13B (54.8%) by 5.3 percentage points. On GSM8k math reasoning, Mistral 7B scores 52.2% (vs 28.7% for Llama 2 13B), nearly doubling mathematical problem-solving performance without specialized fine-tuning.'
          },
          {
            type: 'callout',
            text: 'HumanEval Benchmark: Mistral 7B achieves 30.5% pass@1 on HumanEval, outperforming CodeLlama 7B (30.1%) while preserving general natural language fluency.'
          }
        ]
      },
      {
        pageNumber: 4,
        title: '4. Instruction Tuning & Chat Specialization',
        sectionHeader: 'SECTION 4 • INSTRUCTION MODELING (MISTRAL-7B-INSTRUCT)',
        content: [
          {
            type: 'heading',
            text: '4. Mistral 7B Instruct: Alignment & Safety'
          },
          {
            type: 'paragraph',
            text: 'To demonstrate the instruction-following capabilities of Mistral 7B, we fine-tuned it on instruction datasets using Direct Preference Optimization (DPO) and supervised alignment. Mistral 7B Instruct outperforms all tested 7B and 13B instruction models on MT-Bench.'
          },
          {
            type: 'paragraph',
            text: 'System prompt guardrails allow developers to enforce ethical guidelines without sacrificing helpfulness or introducing aggressive refusals on benign topics.'
          }
        ]
      },
      {
        pageNumber: 5,
        title: '5. Conclusion & Open Release',
        sectionHeader: 'SECTION 5 • REPRODUCIBILITY & APACHE 2.0 RELEASE',
        content: [
          {
            type: 'heading',
            text: '5. Conclusion & Release Information'
          },
          {
            type: 'paragraph',
            text: 'Mistral 7B represents a step forward in foundation model efficiency. By combining sliding window attention, grouped-query attention, and robust tokenization, we provide a versatile, open-weight baseline released under the Apache 2.0 license.'
          }
        ]
      }
    ],
    chunks: [
      {
        id: 'chunk-swa-mem',
        citationId: 1,
        paragraphIndex: 1,
        chunkTitle: 'Sliding Window Attention (SWA) Memory Horizon',
        confidence: 0.96,
        text: 'Sliding Window Attention (SWA) limits the attention span of each token to a local window of size W. In a network with L layers, the theoretical receptive field reaches up to L × W tokens (131k tokens), reducing KV cache footprint by 2.4x.',
        boundingBox: {
          top: 24,
          left: 6,
          width: 88,
          height: 18
        }
      },
      {
        id: 'chunk-gqa-speed',
        citationId: 2,
        paragraphIndex: 3,
        chunkTitle: 'Grouped-Query Attention (GQA) & Latency',
        confidence: 0.92,
        text: 'Mistral 7B incorporates Grouped-Query Attention (GQA) with 8 KV heads for 32 query heads (4:1 ratio), yielding 2.8x throughput increase on 80GB A100 GPUs.',
        boundingBox: {
          top: 56,
          left: 6,
          width: 88,
          height: 16
        }
      },
      {
        id: 'chunk-benchmarks',
        citationId: 3,
        paragraphIndex: 2,
        chunkTitle: 'MMLU & GSM8k Performance vs Llama 2',
        confidence: 0.94,
        text: 'On MMLU, Mistral 7B achieves 60.1% (vs 54.8% for Llama 2 13B). On GSM8k, Mistral 7B scores 52.2% vs 28.7% for Llama 2 13B, nearly doubling mathematical accuracy.',
        boundingBox: {
          top: 42,
          left: 6,
          width: 88,
          height: 17
        }
      }
    ]
  },
  {
    id: 'rag-lewis-2020',
    title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    filename: 'rag-lewis-2020.pdf',
    totalPages: 4,
    fileType: 'pdf',
    uploadedAt: 'Yesterday at 04:15 PM',
    fileSize: '1.8 MB',
    category: 'Information Retrieval & AI',
    authors: 'Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, et al.',
    abstract: 'Large pre-trained language models store factual knowledge in their parameters, but struggle to access precise non-parametric knowledge, hallucinate, and cannot update memories dynamically. We propose RAG models combining pre-trained parametric memory (BART/T5) with a non-parametric vector index of Wikipedia (DPR) retrieved via Maximum Inner Product Search.',
    pages: [
      {
        pageNumber: 1,
        title: '1. Introduction to RAG Architecture',
        sectionHeader: 'NEURIPS 2020 • ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS',
        content: [
          {
            type: 'heading',
            text: 'Retrieval-Augmented Generation (RAG) Architecture'
          },
          {
            type: 'callout',
            text: 'Abstract — We explore general-purpose fine-tuning recipes for Retrieval-Augmented Generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation.'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 4,
            chunkMeta: {
              id: 'chunk-rag-foundations',
              confidence: 0.98,
              label: 'Parametric vs Non-Parametric Memory Synergy'
            },
            text: 'RAG models build a hybrid intelligence paradigm: a pre-trained neural sequence-to-sequence model acts as the parametric generator, while a dense neural index over Wikipedia or custom vector embeddings acts as the non-parametric memory. This separation decouples reasoning capability from static factual memory, allowing real-time index updates without retraining.'
          }
        ]
      },
      {
        pageNumber: 2,
        title: '2. Dense Passage Retrieval (DPR) & FAISS Indexing',
        sectionHeader: 'SECTION 2 • DENSE RETRIEVAL MECHANICS & VECTOR SEARCH',
        content: [
          {
            type: 'heading',
            text: '2. Dense Passage Retrieval (DPR)'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 5,
            chunkMeta: {
              id: 'chunk-dpr-mips',
              confidence: 0.95,
              label: 'DPR & Maximum Inner Product Search (MIPS)'
            },
            text: 'The non-parametric retriever uses Dense Passage Retrieval (DPR) bi-encoders to map queries and document chunks into a shared 768-dimensional embedding space. Retrieval is executed via Maximum Inner Product Search (MIPS) using FAISS indices (HNSW or IVFFlat), fetching top-k relevant passages (typically k = 5 to 10) in sub-15ms latency.'
          },
          {
            type: 'equation',
            text: 'p_η(z|x) ∝ exp(E_q(x)^T E_d(z))   where E_q and E_d are BERT bi-encoders'
          },
          {
            type: 'subheading',
            text: '2.1 RAG-Sequence vs RAG-Token Formulation'
          },
          {
            type: 'paragraph',
            text: 'In RAG-Sequence, the model samples one document to generate the entire response sequence. In RAG-Token, the model marginalizes across different retrieved documents at each individual generated token step, enabling rich synthesis from multiple distinct sources.'
          }
        ]
      },
      {
        pageNumber: 3,
        title: '3. Open-Domain QA & Hallucination Mitigation',
        sectionHeader: 'SECTION 3 • OPEN-DOMAIN QA & FACTUAL ACCURACY',
        content: [
          {
            type: 'heading',
            text: '3. Benchmarks on Natural Questions & TriviaQA'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 6,
            chunkMeta: {
              id: 'chunk-hallucination',
              confidence: 0.93,
              label: 'Hallucination Reduction in Open-Domain QA'
            },
            text: 'On the Natural Questions benchmark, RAG achieves 44.5% Exact Match (EM), setting state-of-the-art results over closed-book models. Crucially, manual human evaluations demonstrate a 68% reduction in hallucinated or fabricated factual claims compared to standalone parametric generators.'
          }
        ]
      },
      {
        pageNumber: 4,
        title: '4. Summary & Implications for Enterprise AI',
        sectionHeader: 'SECTION 4 • CONCLUSION & FUTURE WORK',
        content: [
          {
            type: 'heading',
            text: '4. Conclusion'
          },
          {
            type: 'paragraph',
            text: 'Retrieval-Augmented Generation provides an interpretable, auditable foundation for generative AI. By retaining explicit citation trails back to source documents, users can verify model assertions against retrieved ground truth.'
          }
        ]
      }
    ],
    chunks: [
      {
        id: 'chunk-rag-foundations',
        citationId: 4,
        paragraphIndex: 1,
        chunkTitle: 'Parametric + Non-Parametric Memory Synergy',
        confidence: 0.98,
        text: 'RAG decouples reasoning from factual memory by pairing a pre-trained generator with a dense vector index over external documents, enabling factual updates without retraining.',
        boundingBox: {
          top: 38,
          left: 6,
          width: 88,
          height: 20
        }
      },
      {
        id: 'chunk-dpr-mips',
        citationId: 5,
        paragraphIndex: 1,
        chunkTitle: 'Dense Passage Retrieval (DPR) & MIPS',
        confidence: 0.95,
        text: 'Uses DPR bi-encoders to map queries and chunks into shared 768-d space, executing sub-15ms retrieval via FAISS Maximum Inner Product Search.',
        boundingBox: {
          top: 22,
          left: 6,
          width: 88,
          height: 20
        }
      },
      {
        id: 'chunk-hallucination',
        citationId: 6,
        paragraphIndex: 1,
        chunkTitle: 'Hallucination Reduction & Exact Match Gains',
        confidence: 0.93,
        text: 'Achieves 44.5% Exact Match on Natural Questions and exhibits a 68% reduction in factual hallucinations compared to purely parametric models.',
        boundingBox: {
          top: 24,
          left: 6,
          width: 88,
          height: 19
        }
      }
    ]
  },
  {
    id: 'mistral-large-2024',
    title: 'Mistral Large 2: State-of-the-Art Frontier Reasoning',
    filename: 'mistral-large-2024.pdf',
    totalPages: 4,
    fileType: 'pdf',
    uploadedAt: '3 days ago',
    fileSize: '3.1 MB',
    category: 'Frontier Models',
    authors: 'Mistral AI Frontier Research Group',
    abstract: 'Mistral Large 2 (123B) is built for complex multilingual reasoning, mathematical synthesis, and high-precision function calling across 128,000 token context windows. It matches GPT-4o and Claude 3.5 Sonnet on code and analytical benchmarks.',
    pages: [
      {
        pageNumber: 1,
        title: '1. Architecture & 128k Context Window',
        sectionHeader: 'MISTRAL LARGE 2 TECHNICAL REPORT • JULY 2024',
        content: [
          {
            type: 'heading',
            text: 'Mistral Large 2: Advanced Reasoning & Context Scaling'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 7,
            chunkMeta: {
              id: 'chunk-large-context',
              confidence: 0.97,
              label: '128k Long-Context Needle-In-A-Haystack'
            },
            text: 'Mistral Large 2 features a 128k token context window with native support for 80+ programming and natural languages. In Needle-In-A-Haystack evaluations, the model maintains 99.4% retrieval accuracy across all token depths from 1k to 128k tokens.'
          }
        ]
      },
      {
        pageNumber: 2,
        title: '2. Code Generation & Tool Use',
        sectionHeader: 'SECTION 2 • CODING CAPABILITIES & FUNCTION CALLING',
        content: [
          {
            type: 'heading',
            text: '2. Function Calling & Agentic Workflows'
          },
          {
            type: 'paragraph',
            isChunk: true,
            citationId: 8,
            chunkMeta: {
              id: 'chunk-tool-calling',
              confidence: 0.95,
              label: 'Strict JSON Schema & Parallel Tool Calling'
            },
            text: 'Mistral Large 2 is tuned for constrained decoding and parallel tool execution. It achieves 84.0% on the Berkeley Function-Calling Benchmark (BFCL), reliably invoking complex multi-step APIs with precise parameter validation.'
          }
        ]
      },
      {
        pageNumber: 3,
        title: '3. Multilingual Performance & Cost Efficiency',
        sectionHeader: 'SECTION 3 • BENCHMARKS & DEPLOYMENT',
        content: [
          {
            type: 'heading',
            text: '3. Multilingual Reasoning & Token Economics'
          },
          {
            type: 'paragraph',
            text: 'With high token throughput and competitive pricing, Mistral Large 2 delivers enterprise-grade reasoning at a fraction of the inference latency of previous generation frontier models.'
          }
        ]
      },
      {
        pageNumber: 4,
        title: '4. Summary',
        sectionHeader: 'SECTION 4 • CONCLUSION',
        content: [
          {
            type: 'heading',
            text: '4. Conclusion'
          },
          {
            type: 'paragraph',
            text: 'Mistral Large 2 represents Mistral AI’s flagship model for mission-critical reasoning, RAG architectures, and complex code refactoring.'
          }
        ]
      }
    ],
    chunks: [
      {
        id: 'chunk-large-context',
        citationId: 7,
        paragraphIndex: 1,
        chunkTitle: '128k Needle-in-a-Haystack Retrieval',
        confidence: 0.97,
        text: 'Maintains 99.4% retrieval accuracy across 128k token context lengths in Needle-in-a-Haystack benchmarks.',
        boundingBox: {
          top: 30,
          left: 6,
          width: 88,
          height: 18
        }
      },
      {
        id: 'chunk-tool-calling',
        citationId: 8,
        paragraphIndex: 1,
        chunkTitle: 'Parallel Function Calling & JSON Schema',
        confidence: 0.95,
        text: 'Achieves 84.0% on the Berkeley Function-Calling Benchmark, reliably executing multi-turn tool calling.',
        boundingBox: {
          top: 26,
          left: 6,
          width: 88,
          height: 18
        }
      }
    ]
  }
];

export const PRESET_PROMPTS: PresetPrompt[] = [
  {
    id: 'prompt-1',
    title: 'Sliding Window Attention',
    subtitle: 'Mistral 7B KV Cache Optimization',
    query: 'How does Sliding Window Attention (SWA) reduce memory footprint in Mistral 7B while preserving long context?',
    targetDocId: 'mistral-7b-v01',
    icon: 'Cpu',
    category: 'Architecture'
  },
  {
    id: 'prompt-2',
    title: 'RAG Core Foundations',
    subtitle: 'Lewis et al. Non-Parametric Memory',
    query: 'What is the fundamental difference between parametric and non-parametric memory in RAG architectures?',
    targetDocId: 'rag-lewis-2020',
    icon: 'Layers',
    category: 'Information Retrieval'
  },
  {
    id: 'prompt-3',
    title: 'Grouped-Query Attention',
    subtitle: 'Decoding Throughput & 8k Latency',
    query: 'Explain how Grouped-Query Attention (GQA) with 8 KV heads improves inference throughput on A100 GPUs.',
    targetDocId: 'mistral-7b-v01',
    icon: 'Zap',
    category: 'Performance'
  },
  {
    id: 'prompt-4',
    title: 'Hallucination Mitigation',
    subtitle: 'Exact Match & Faithfulness Benchmarks',
    query: 'How does grounding LLMs with Dense Passage Retrieval (DPR) reduce factual hallucinations?',
    targetDocId: 'rag-lewis-2020',
    icon: 'ShieldCheck',
    category: 'Accuracy'
  }
];

export const INITIAL_CITATIONS: Record<number, Citation> = {
  1: {
    id: 1,
    docId: 'mistral-7b-v01',
    docTitle: 'Mistral 7B: Efficient Open Weight Foundation Model',
    page: 2,
    snippet: 'Sliding Window Attention (SWA) limits attention span to W = 4096 tokens, providing theoretical receptive field of ~131k tokens across 32 layers while reducing KV cache memory by 2.4x.',
    chunkTitle: 'Sliding Window Attention & KV Cache Reduction',
    confidence: 0.96,
    score: 0.912,
    boundingBox: { top: 22, left: 6, width: 88, height: 18 }
  },
  2: {
    id: 2,
    docId: 'mistral-7b-v01',
    docTitle: 'Mistral 7B: Efficient Open Weight Foundation Model',
    page: 2,
    snippet: 'GQA uses 8 KV heads for 32 query heads (4:1 ratio), shrinking KV cache during decoding and boosting inference throughput by 2.8x on 80GB A100 GPUs.',
    chunkTitle: 'Grouped-Query Attention (GQA) & Latency',
    confidence: 0.92,
    score: 0.884,
    boundingBox: { top: 54, left: 6, width: 88, height: 16 }
  },
  3: {
    id: 3,
    docId: 'mistral-7b-v01',
    docTitle: 'Mistral 7B: Efficient Open Weight Foundation Model',
    page: 3,
    snippet: 'Mistral 7B scores 60.1% on MMLU (surpassing Llama 2 13B at 54.8%) and 52.2% on GSM8k math reasoning.',
    chunkTitle: 'MMLU & GSM8k Performance vs Llama 2',
    confidence: 0.94,
    score: 0.895,
    boundingBox: { top: 40, left: 6, width: 88, height: 17 }
  },
  4: {
    id: 4,
    docId: 'rag-lewis-2020',
    docTitle: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    page: 1,
    snippet: 'RAG pairs a pre-trained seq2seq parametric generator with a non-parametric neural vector index over external passages, decoupling reasoning from static memorization.',
    chunkTitle: 'Parametric + Non-Parametric Memory Synergy',
    confidence: 0.98,
    score: 0.941,
    boundingBox: { top: 36, left: 6, width: 88, height: 20 }
  },
  5: {
    id: 5,
    docId: 'rag-lewis-2020',
    docTitle: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    page: 2,
    snippet: 'DPR bi-encoders map queries and document chunks into shared 768-d space, executing sub-15ms retrieval via FAISS Maximum Inner Product Search (MIPS).',
    chunkTitle: 'Dense Passage Retrieval (DPR) & MIPS',
    confidence: 0.95,
    score: 0.908,
    boundingBox: { top: 20, left: 6, width: 88, height: 20 }
  },
  6: {
    id: 6,
    docId: 'rag-lewis-2020',
    docTitle: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    page: 3,
    snippet: 'RAG achieves 44.5% Exact Match on Natural Questions and provides a 68% reduction in hallucinated claims compared to standard parametric models.',
    chunkTitle: 'Hallucination Reduction in Open-Domain QA',
    confidence: 0.93,
    score: 0.887,
    boundingBox: { top: 22, left: 6, width: 88, height: 19 }
  },
  7: {
    id: 7,
    docId: 'mistral-large-2024',
    docTitle: 'Mistral Large 2: State-of-the-Art Frontier Reasoning',
    page: 1,
    snippet: 'Features 128k token context window with 99.4% retrieval accuracy across all token depths in Needle-In-A-Haystack evaluations.',
    chunkTitle: '128k Long-Context Needle-In-A-Haystack',
    confidence: 0.97,
    score: 0.929,
    boundingBox: { top: 28, left: 6, width: 88, height: 18 }
  },
  8: {
    id: 8,
    docId: 'mistral-large-2024',
    docTitle: 'Mistral Large 2: State-of-the-Art Frontier Reasoning',
    page: 2,
    snippet: 'Scores 84.0% on the Berkeley Function-Calling Benchmark, executing multi-turn tool calling with strict schema enforcement.',
    chunkTitle: 'Parallel Function Calling & JSON Schema',
    confidence: 0.95,
    score: 0.915,
    boundingBox: { top: 24, left: 6, width: 88, height: 18 }
  }
};

export const INITIAL_MESSAGES = [
  {
    id: 'msg-welcome',
    role: 'assistant' as const,
    content: `Welcome to **RAG Document Studio**! I am your research assistant powered by dense vector retrieval and neural chunk grounding.

Ask any question about your indexed papers and knowledge bases. When I cite evidence, click on the highlighted badges like **[1]** or **[2]** to immediately inspect the source document, page number, and vector bounding box in the right pane.

**Try asking:**
- *"How does Sliding Window Attention reduce memory in Mistral 7B?"*
- *"What are the core components of RAG non-parametric memory?"*
- *"Compare Multi-Head Attention with Grouped-Query Attention throughput."*`,
    timestamp: 'Just now',
    model: 'Mistral Large 2',
    citations: [
      INITIAL_CITATIONS[1],
      INITIAL_CITATIONS[2]
    ],
    retrievalLatencyMs: 240,
    tokensCount: 168
  }
];
