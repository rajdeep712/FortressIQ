// Minimal no-router URL helpers for conversation slugs (/c/<chat_id>).
// chat_id is the same id the backend uses as the LangGraph Postgres thread_id.

export function conversationPath(chatId: string): string {
  return `/c/${encodeURIComponent(chatId)}`;
}

export function currentConversationSlug(): string | null {
  if (typeof window === 'undefined') return null;
  const m = window.location.pathname.match(/^\/c\/([a-zA-Z0-9_-]+)\/?$/);
  return m ? m[1] : null;
}

export function isHomeRoute(): boolean {
  const p = window.location.pathname;
  return p === '/' || p === '';
}

export function goHome(): void {
  if (!isHomeRoute()) {
    window.history.pushState({}, '', '/');
  }
}

export function goToConversation(chatId: string): void {
  window.history.pushState({}, '', conversationPath(chatId));
}

export function setConversationUrl(chatId: string): void {
  // Replaces the current entry (used when the first query creates a chat so
  // Back skips a duplicate home page).
  window.history.replaceState({}, '', conversationPath(chatId));
}