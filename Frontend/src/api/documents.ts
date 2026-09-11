import { api, API_BASE, ApiError } from './client';
import { RAGDocument } from '../types';

export interface DocumentUploadResponse {
  doc_id: string;
  user_id: string;
  filename: string;
  mime_type: string;
  size: number;
  sha256: string;
  s3_key: string;
  status: string;
  created_at: string;
}

export interface DocumentStatusResponse {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  s3_key: string;
  created_at: string;
}

export interface DocumentRetryResponse {
  doc_id: string;
  status: string;
  message: string;
}

export interface DocumentSummary {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  file_size: number;
  mime_type: string;
  created_at: string;
}

// Terminal ingestion states (see backend Document.status).
export const DOC_READY_STATUS = 'COMPLETED';
export const DOC_FAILED_STATUS = 'FAILED';
export const DOC_ACTIVE_STATUSES = ['UPLOADED', 'PROCESSING', 'PARSED', 'CHUNKED', 'EMBEDDED'];

export function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  return api<DocumentUploadResponse>('/documents/upload', {
    method: 'POST',
    formData,
  });
}

export function getDocument(docId: string): Promise<DocumentStatusResponse> {
  return api<DocumentStatusResponse>(`/documents/${encodeURIComponent(docId)}`);
}

export function retryDocument(docId: string): Promise<DocumentRetryResponse> {
  return api<DocumentRetryResponse>(`/documents/${encodeURIComponent(docId)}/retry`, {
    method: 'POST',
  });
}

export function listDocuments(): Promise<DocumentSummary[]> {
  return api<DocumentSummary[]>('/documents');
}

const formatBytes = (bytes: number): string => {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
};

const getFileType = (name: string): RAGDocument['fileType'] => {
  const ext = name.split('.').pop()?.toLowerCase();
  if (ext === 'pdf') return 'pdf';
  if (ext === 'doc' || ext === 'docx' || ext === 'odt') return 'doc';
  return 'txt';
};

export function toRagDocument(summary: DocumentSummary): RAGDocument {
  return {
    id: summary.doc_id,
    title: summary.filename.replace(/\.[^/.]+$/, ''),
    filename: summary.filename,
    // Page viewer is implemented later; restored uploads show as ingested and
    // awaiting retrieval with no consumer-facing pages yet.
    totalPages: summary.chunk_count || 0,
    fileType: getFileType(summary.filename),
    uploadedAt: 'Just now',
    fileSize: formatBytes(summary.file_size),
    category: 'Custom Upload',
    abstract: 'Real uploaded document, indexed and available for retrieval.',
    pages: [],
    chunks: [],
  };
}

export interface DocumentWatchHandlers {
  onStatus: (status: DocumentStatusResponse) => void;
  onError?: (error: Error) => void;
}

export interface DocumentWatchHandle {
  close: () => void;
}

interface SseFrame {
  event: string | null;
  data: string | null;
}

function parseSseFrame(raw: string): SseFrame {
  let event: string | null = null;
  let data: string | null = null;
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data = line.slice('data:'.length).trim();
  }
  return { event, data };
}

const MAX_RECONNECT_DELAY_MS = 15000;
const TERMINAL_STATUSES = [DOC_READY_STATUS, DOC_FAILED_STATUS];

// Streams ingestion status updates for a document via Server-Sent Events.
// Reconnects with exponential backoff after non-terminal drops; closes
// automatically once a terminal status has been observed. Returns a handle
// whose `close()` tears the stream down.
export function watchDocumentStatus(
  docId: string,
  handlers: DocumentWatchHandlers,
): DocumentWatchHandle {
  const controller = new AbortController();
  let closed = false;
  let terminal = false;
  let retries = 0;

  const scheduleReconnect = () => {
    if (closed || terminal) return;
    const delay = Math.min(1000 * 2 ** retries, MAX_RECONNECT_DELAY_MS);
    window.setTimeout(() => {
      if (!closed && !terminal) void open();
    }, delay);
  };

  const open = async (allowAuthRefresh = true) => {
    if (closed) return;

    let res: Response;
    try {
      res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}/events`, {
        credentials: 'include',
        headers: {
          Accept: 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        signal: controller.signal,
      });
    } catch {
      if (closed) return;
      retries += 1;
      scheduleReconnect();
      return;
    }

    if (res.status === 401 && allowAuthRefresh) {
      try {
        await fetch(`${API_BASE}/auth/refresh`, { method: 'POST', credentials: 'include' });
      } catch {
        // Fall through — the retry below will surface the auth error.
      }
      await open(false);
      return;
    }

    if (!res.ok || !res.body) {
      if (!closed) handlers.onError?.(new ApiError(res.status, null));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (!closed) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';
        for (const raw of frames) {
          const { event, data } = parseSseFrame(raw);
          if (
            (event === 'connected' || event === 'status') &&
            data !== null
          ) {
            let payload: DocumentStatusResponse;
            try {
              payload = JSON.parse(data) as DocumentStatusResponse;
            } catch {
              continue;
            }
            retries = 0;
            if (TERMINAL_STATUSES.includes(payload.status)) {
              terminal = true;
            }
            handlers.onStatus(payload);
          }
        }
      }
    } catch {
      if (closed) return;
      handlers.onError?.(new Error('Document event stream was interrupted.'));
      scheduleReconnect();
      return;
    }

    scheduleReconnect();
  };

  void open();

  return {
    close: () => {
      closed = true;
      controller.abort();
    },
  };
}