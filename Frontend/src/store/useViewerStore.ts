import { create } from 'zustand';
import { Citation, Message, RAGDocument } from '../types';
import type { AuthUser } from '../api/auth';
import type { BackendConversation } from '../api/chat';

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

  // Per-user chat retrieval scope: null means "all documents".
  selectedDocIds: string[] | null;

  // Bootstrap / loading flags (drive skeletons instead of guest marketing UI)
  isSessionRestoring: boolean;
  isLoadingDocuments: boolean;
  isLoadingMessages: boolean;

  // Conversation history (backend chat)
  activeChatId: string | null;
  conversations: BackendConversation[];
  isLoadingConversations: boolean;
  
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
  setMessages: (messages: Message[]) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  clearChat: () => void;

  // Conversation history actions
  setActiveChatId: (chatId: string | null) => void;
  setConversations: (conversations: BackendConversation[]) => void;
  setIsLoadingConversations: (v: boolean) => void;
  appendStreamToken: (messageId: string, token: string) => void;
  finalizeAssistant: (messageId: string, payload: {
    content?: string;
    citations?: Citation[];
    model?: string;
    retrievalLatencyMs?: number;
  }) => void;

  uploadDocument: (newDoc: RAGDocument) => void;
  setDocuments: (documents: RAGDocument[]) => void;
  setSelectedDocIds: (docIds: string[] | null) => void;

  // Auth state (guest by default; wired to the cookie backend)
  user: AuthUser | null;
  isAuthenticated: boolean;
  showAuthModal: boolean;
  setUser: (user: AuthUser | null) => void;
  setAuthenticated: (v: boolean) => void;
  setShowAuthModal: (v: boolean) => void;
  setSessionRestoring: (v: boolean) => void;
  setIsLoadingDocuments: (v: boolean) => void;
  setIsLoadingMessages: (v: boolean) => void;
}

export const useViewerStore = create<ViewerState>((set, get) => ({
  activeDocumentId: null,
  activePage: 1,
  isViewerOpen: false,

  activeCitationId: null,
  activeHighlight: null,
  pulseTrigger: 1,

  documents: [],
  messages: [],
  isGenerating: false,
  selectedModel: 'Mistral Large 2',
  selectedDocIds: null,

  activeChatId: null,
  conversations: [],
  isLoadingConversations: false,

  zoomLevel: 100,
  searchQueryInDoc: '',
  isOutlineOpen: false,
  isSessionRestoring: true,
  isLoadingDocuments: false,
  isLoadingMessages: false,

  user: null,
  isAuthenticated: false,
  showAuthModal: false,

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
  setMessages: (messages) => set({ messages }),
  setIsGenerating: (isGenerating) => set({ isGenerating }),

  clearChat: () => {
    set({
      messages: [],
      activeChatId: null,
      activeCitationId: null,
      activeHighlight: null
    });
  },

  setActiveChatId: (chatId) => set({ activeChatId: chatId }),
  setConversations: (conversations) => set({ conversations }),
  setIsLoadingConversations: (isLoadingConversations) => set({ isLoadingConversations }),

  appendStreamToken: (messageId, token) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId
          ? { ...m, content: m.content + token }
          : m
      )
    })),

  finalizeAssistant: (messageId, payload) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId
          ? {
              ...m,
              content: payload.content !== undefined ? payload.content : m.content,
              citations: payload.citations !== undefined ? payload.citations : m.citations,
              model: payload.model ?? m.model,
              retrievalLatencyMs: payload.retrievalLatencyMs ?? m.retrievalLatencyMs
            }
          : m
      ),
      isGenerating: false
    })),

  uploadDocument: (newDoc) => {
    set((state) => ({
      documents: [newDoc, ...state.documents],
      activeDocumentId: newDoc.id,
      activePage: 1,
      activeCitationId: null,
      activeHighlight: null
    }));
  },

  setDocuments: (documents) => set({ documents }),

  setSelectedDocIds: (docIds) => set({ selectedDocIds: docIds }),

  setAuthenticated: (v) => set({ isAuthenticated: v }),
  setShowAuthModal: (v) => set({ showAuthModal: v }),
  setUser: (user) =>
    set((state) => ({
      user,
      isAuthenticated: Boolean(user),
      // Documents are user-scoped; drop them on logout so a fresh sign-in
      // never sees the previous account's papers.
      documents: user ? state.documents : [],
      selectedDocIds: user ? state.selectedDocIds : null,
      isLoadingDocuments: false,
      isLoadingMessages: false
    })),

  setSessionRestoring: (isSessionRestoring) => set({ isSessionRestoring }),
  setIsLoadingDocuments: (isLoadingDocuments) => set({ isLoadingDocuments }),
  setIsLoadingMessages: (isLoadingMessages) => set({ isLoadingMessages })
}));
