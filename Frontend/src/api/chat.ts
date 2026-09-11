// Chat API client: SSE streaming chat + conversation/message management.
//
// The chat endpoint streams Server-Sent Events over a POST (fetch +
// ReadableStream). Cookies (httpOnly) carry auth, so credentials are included.

import { API_BASE } from './client';
import type { Citation } from '../types';

export interface BackendCitation {
  id: number;
  doc_id: string;
  doc_title?: string;
  page?: number;
  snippet?: string;
  chunk_title?: string;
  confidence?: number;
  score?: number;
  bounding_box?: { top: number; left: number; width: number; height: number };
}

export interface BackendChatMessage {
  message_id: string;
  chat_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: BackendCitation[];
  created_at: string;
}

export interface BackendConversation {
  chat_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

// The `meta` SSE frame (emitted before the first token).
export interface ChatMeta {
  chat_id: string;
  route_decision: string | null;
  route_reason: string | null;
  num_message_turns: number;
  retrieval_latency_ms: number;
  has_grounding_context: boolean;
  model?: string;
}

// The `done` SSE frame (final answer + citations).
export interface ChatDone {
  answer: string;
  citations: BackendCitation[];
  error?: string;
}

export interface StreamCallbacks {
  onMeta?: (meta: ChatMeta) => void;
  onToken?: (token: string) => void;
  onDone?: (done: ChatDone) => void;
}

export interface StreamRequest {
  message: string;
  chatId?: string | null;
  selectedDocIds?: string[] | null;
  signal?: AbortSignal;
}

export function toFrontendCitation(c: BackendCitation): Citation {
  const b = c.bounding_box ?? { top: 0, left: 0, width: 100, height: 100 };
  return {
    id: c.id,
    docId: c.doc_id,
    docTitle: c.doc_title || 'Source document',
    page: c.page || 1,
    snippet: c.snippet || '',
    chunkTitle: c.chunk_title || 'Document passage',
    confidence: c.confidence ?? 0,
    score: c.score ?? 0,
    boundingBox: {
      top: b.top ?? 0,
      left: b.left ?? 0,
      width: b.width ?? 100,
      height: b.height ?? 100,
    },
  };
}

export async function streamChat(
  req: StreamRequest,
  cb: StreamCallbacks,
): Promise<ChatDone> {
  const res = await fetch(`${API_BASE}/chat/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    signal: req.signal,
    body: JSON.stringify({
      message: req.message,
      chat_id: req.chatId ?? null,
      selected_doc_ids: req.selectedDocIds ?? null,
    }),
  });

  if (!res.ok) {
    let detail = `Chat request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data && typeof data === 'object' && 'detail' in data) {
        detail = String(data.detail);
      }
    } catch {
      // ignore parse failures
    }
    throw new Error(detail);
  }

  if (!res.body) {
    throw new Error('Chat stream unavailable.');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // Fulfilled once the `done` frame arrives (or rejected on error).
  let doneResolve!: (value: ChatDone) => void;
  let doneReject!: (reason: unknown) => void;
  const donePromise = new Promise<ChatDone>((resolve, reject) => {
    doneResolve = resolve;
    doneReject = reject;
  });

  const handleFrame = (event: string, data: string | null) => {
    if (!data) return;
    if (event === 'token') {
      const parsed = JSON.parse(data);
      cb.onToken?.(parsed.token ?? '');
    } else if (event === 'meta') {
      cb.onMeta?.(JSON.parse(data) as ChatMeta);
    } else if (event === 'done') {
      const parsed = JSON.parse(data) as ChatDone;
      if (parsed.error) {
        doneReject(new Error(parsed.error));
        return;
      }
      cb.onDone?.(parsed);
      doneResolve(parsed);
    }
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by blank lines.
    let sepIndex: number;
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      let event = 'message';
      let data: string | null = null;
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) {
          event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          data = (data ? data + '\n' : '') + line.slice(5).trim();
        }
      }
      if (data !== null) {
        try {
          handleFrame(event, data);
        } catch (err) {
          doneReject(err);
          return donePromise;
        }
      }
    }
  }

  return donePromise;
}

export async function listConversations(): Promise<BackendConversation[]> {
  const res = await fetch(`${API_BASE}/chat/conversations`, { credentials: 'include' });
  if (!res.ok) throw new Error('Could not load conversations.');
  return (await res.json()) as BackendConversation[];
}

export async function getMessages(chatId: string): Promise<BackendChatMessage[]> {
  const res = await fetch(`${API_BASE}/chat/${encodeURIComponent(chatId)}/messages`, {
    credentials: 'include',
  });
  if (!res.ok) throw new Error('Could not load messages.');
  return (await res.json()) as BackendChatMessage[];
}

export async function deleteConversation(chatId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/${encodeURIComponent(chatId)}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (!res.ok) throw new Error('Could not delete conversation.');
}
