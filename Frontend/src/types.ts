export interface Citation {
  id: number;
  docId: string;
  docTitle: string;
  page: number;
  snippet: string;
  chunkTitle: string;
  confidence: number;
  score: number;
  boundingBox: {
    top: number; // percentage from top of page (e.g., 24%)
    left: number; // percentage from left (e.g., 8%)
    width: number; // percentage width (e.g., 84%)
    height: number; // percentage height (e.g., 18%)
  };
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  citations?: Citation[];
  model?: string;
  retrievalLatencyMs?: number;
  tokensCount?: number;
}

export interface DocumentChunk {
  id: string;
  citationId?: number;
  paragraphIndex: number;
  text: string;
  boundingBox: {
    top: number;
    left: number;
    width: number;
    height: number;
  };
  chunkTitle: string;
  confidence: number;
}

export interface DocumentPage {
  pageNumber: number;
  title?: string;
  sectionHeader?: string;
  content: {
    type: 'heading' | 'subheading' | 'paragraph' | 'callout' | 'equation' | 'table' | 'bullet-list';
    text?: string;
    items?: string[];
    caption?: string;
    isChunk?: boolean;
    citationId?: number;
    chunkMeta?: {
      id: string;
      confidence: number;
      label: string;
    };
  }[];
}

export interface RAGDocument {
  id: string;
  title: string;
  filename: string;
  totalPages: number;
  fileType: 'pdf' | 'doc' | 'txt';
  uploadedAt: string;
  fileSize: string;
  category: string;
  authors?: string;
  abstract: string;
  pages: DocumentPage[];
  chunks: DocumentChunk[];
}

export interface PresetPrompt {
  id: string;
  title: string;
  subtitle: string;
  query: string;
  targetDocId: string;
  icon: string;
  category: string;
}
