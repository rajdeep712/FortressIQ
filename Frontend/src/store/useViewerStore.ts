import { create } from 'zustand';
import { Citation, Message, RAGDocument } from '../types';
import { INITIAL_CITATIONS, INITIAL_MESSAGES, MOCK_DOCUMENTS } from '../data/mockDocuments';

interface HighlightBox {
  top: number;
  left: number;
  width: number;
  height: number;
  text?: string;
  chunkTitle?: string;
  page: number;
  citationId?: number;
  confidence?: number;
}

interface ViewerState {
  // Core store requirements
  activeDocumentId: string | null;
  activePage: number;
  isViewerOpen: boolean;

  // Rich RAG interaction states
  activeCitationId: number | null;
  activeHighlight: HighlightBox | null;
  pulseTrigger: number; // Increment to trigger pulse animation
  
  // Document and Chat state
  documents: RAGDocument[];
  messages: Message[];
  isGenerating: boolean;
  selectedModel: string;
  
  // Viewer presentation state
  zoomLevel: number;
  searchQueryInDoc: string;
  isOutlineOpen: boolean;

  // Actions
  setActiveDocumentId: (docId: string | null, page?: number) => void;
  setActivePage: (page: number) => void;
  setViewerOpen: (open: boolean) => void;

  jumpToCitation: (citation: Citation) => void;
  clearActiveCitation: () => void;

  setZoomLevel: (zoom: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  setSearchQueryInDoc: (query: string) => void;
  setIsOutlineOpen: (open: boolean) => void;

  setSelectedModel: (model: string) => void;
  addMessage: (message: Message) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  clearChat: () => void;

  uploadDocument: (newDoc: RAGDocument) => void;
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  // Default to Mistral 7B Paper on page 2 so user immediately sees interactive highlighted RAG chunk
  activeDocumentId: 'mistral-7b-v01',
  activePage: 2,
  isViewerOpen: false,

  activeCitationId: 1,
  activeHighlight: {
    top: 22,
    left: 6,
    width: 88,
    height: 18,
    text: 'Sliding Window Attention (SWA) limits the attention span of each token to a local window of size W. In a network with L layers, the theoretical receptive field reaches up to L × W tokens...',
    chunkTitle: 'Sliding Window Attention & KV Cache Reduction',
    page: 2,
    citationId: 1,
    confidence: 0.96
  },
  pulseTrigger: 1,

  documents: MOCK_DOCUMENTS,
  messages: INITIAL_MESSAGES,
  isGenerating: false,
  selectedModel: 'Mistral Large 2',

  zoomLevel: 100,
  searchQueryInDoc: '',
  isOutlineOpen: false,

  setActiveDocumentId: (docId, page = 1) => {
    set({
      activeDocumentId: docId,
      activePage: page,
      activeCitationId: null,
      activeHighlight: null,
      searchQueryInDoc: ''
    });
  },

  setActivePage: (page) => {
    const { activeDocumentId, documents } = get();
    if (!activeDocumentId) return;
    const doc = documents.find((d) => d.id === activeDocumentId);
    if (!doc) return;
    const clamped = Math.max(1, Math.min(page, doc.totalPages));
    set({ activePage: clamped });
  },

  setViewerOpen: (open) => set({ isViewerOpen: open }),

  jumpToCitation: (citation: Citation) => {
    set((state) => {
      // Find the document to ensure page bounds
      const doc = state.documents.find((d) => d.id === citation.docId);
      const targetPage = doc ? Math.max(1, Math.min(citation.page, doc.totalPages)) : citation.page;

      return {
        activeDocumentId: citation.docId,
        activePage: targetPage,
        activeCitationId: citation.id,
        activeHighlight: {
          top: citation.boundingBox.top,
          left: citation.boundingBox.left,
          width: citation.boundingBox.width,
          height: citation.boundingBox.height,
          text: citation.snippet,
          chunkTitle: citation.chunkTitle,
          page: targetPage,
          citationId: citation.id,
          confidence: citation.confidence
        },
        pulseTrigger: state.pulseTrigger + 1,
        isViewerOpen: true // Open viewer automatically on mobile
      };
    });
  },

  clearActiveCitation: () => {
    set({
      activeCitationId: null,
      activeHighlight: null
    });
  },

  setZoomLevel: (zoom) => {
    const clamped = Math.max(50, Math.min(zoom, 200));
    set({ zoomLevel: clamped });
  },

  zoomIn: () => {
    set((state) => ({
      zoomLevel: Math.min(state.zoomLevel + 15, 200)
    }));
  },

  zoomOut: () => {
    set((state) => ({
      zoomLevel: Math.max(state.zoomLevel - 15, 50)
    }));
  },

  resetZoom: () => set({ zoomLevel: 100 }),

  setSearchQueryInDoc: (query) => set({ searchQueryInDoc: query }),
  setIsOutlineOpen: (open) => set({ isOutlineOpen: open }),

  setSelectedModel: (model) => set({ selectedModel: model }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  setIsGenerating: (isGenerating) => set({ isGenerating }),

  clearChat: () => {
    set({
      messages: INITIAL_MESSAGES,
      activeCitationId: null,
      activeHighlight: null
    });
  },

  uploadDocument: (newDoc) => {
    set((state) => ({
      documents: [newDoc, ...state.documents],
      activeDocumentId: newDoc.id,
      activePage: 1,
      activeCitationId: null,
      activeHighlight: null
    }));
  }
}));
