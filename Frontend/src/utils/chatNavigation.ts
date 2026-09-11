import { useViewerStore } from '../store/useViewerStore';
import { getMessages, toFrontendCitation } from '../api/chat';
import type { BackendChatMessage } from '../api/chat';
import type { Message } from '../types';

export function mapBackendMessage(m: BackendChatMessage): Message {
  return {
    id: m.message_id,
    role: m.role,
    content: m.content,
    timestamp: m.created_at
      ? new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : '',
    citations: m.citations?.map(toFrontendCitation),
  };
}

export async function selectConversation(chatId: string): Promise<void> {
  const s = useViewerStore.getState();
  if (s.activeChatId === chatId) return;
  s.setActiveChatId(chatId);
  s.setIsLoadingMessages(true);
  try {
    const backend = await getMessages(chatId);
    s.setMessages(backend.map(mapBackendMessage));
  } catch {
    s.setMessages([]);
  } finally {
    s.setIsLoadingMessages(false);
  }
}