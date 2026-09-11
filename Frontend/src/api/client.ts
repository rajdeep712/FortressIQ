// Minimal typed API client for the RAG backend.
//
// Cookies (httpOnly accessToken/refreshToken) carry auth, so every request
// includes credentials. On a 401 for an authed endpoint we opportunistically
// call /auth/refresh once (rotating cookies) and retry, matching the Node
// client behavior.

export const API_BASE = '/api/v1';

export class ApiError extends Error {
  status: number;
  detail: string | Record<string, unknown> | null;

  constructor(status: number, detail: string | Record<string, unknown> | null, message?: string) {
    super(message ?? (typeof detail === 'string' ? detail : `Request failed (${status})`));
    this.status = status;
    this.detail = detail;
  }
}

export interface RequestOptions {
  method?: string;
  body?: unknown;
  // Raw FormData (uploads) — set body to a FormData and skip JSON encoding.
  formData?: FormData;
}

async function parseError(res: Response): Promise<ApiError> {
  let detail: string | Record<string, unknown> | null = null;
  try {
    const data = await res.json();
    if (data && typeof data === 'object' && 'detail' in data) {
      detail = data.detail as string | Record<string, unknown>;
    }
  } catch {
    // ignore parse failures
  }
  return new ApiError(res.status, detail);
}

// Single-flight refresh: concurrent 401s share ONE /auth/refresh request so
// the single-use refresh token is never presented twice. Resolves regardless
// of the outcome; callers decide from the retried request's result.
let refreshPromise: Promise<void> | null = null;

function refreshTokens(): Promise<void> {
  if (!refreshPromise) {
    refreshPromise = fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
    })
      .then(() => undefined)
      .catch(() => undefined)
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}

async function doFetch(path: string, options: RequestOptions = {}, attempt = 1): Promise<Response> {
  const method = options.method ?? 'GET';
  const headers: Record<string, string> = {};
  if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    credentials: 'include',
    body:
      options.formData !== undefined
        ? options.formData
        : options.body !== undefined
          ? JSON.stringify(options.body)
          : undefined,
  });

  // Transparently refresh and retry once on an auth failure. Guard against
  // loops by only retrying the first time.
  if (res.status === 401 && attempt === 1) {
    await refreshTokens();
    return doFetch(path, options, attempt + 1);
  }

  return res;
}

export async function api<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const res = await doFetch(path, options);
  if (!res.ok) {
    throw await parseError(res);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export async function apiVoid(path: string, options: RequestOptions = {}): Promise<void> {
  const res = await doFetch(path, options);
  if (!res.ok) {
    throw await parseError(res);
  }
}
