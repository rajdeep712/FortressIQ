// Per-user chat retrieval scope persistence (localStorage).
//
// `null` means "all documents" (the default); an explicit array pins a
// subset. The value is stored under a user-scoped key so switching accounts
// in the same browser never leaks one user's selection to another.

const DOC_SELECTION_PREFIX = 'rag:docSelection:';

export function loadDocSelection(userId: string): string[] | null {
  try {
    const raw = window.localStorage.getItem(DOC_SELECTION_PREFIX + userId);
    if (raw === null) return null;
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as string[]) : null;
  } catch {
    return null;
  }
}

export function saveDocSelection(userId: string, docIds: string[] | null): void {
  try {
    if (docIds === null) {
      window.localStorage.removeItem(DOC_SELECTION_PREFIX + userId);
    } else {
      window.localStorage.setItem(
        DOC_SELECTION_PREFIX + userId,
        JSON.stringify(docIds),
      );
    }
  } catch {
    // Storage unavailable (private mode, quota) — scope just won't persist.
  }
}