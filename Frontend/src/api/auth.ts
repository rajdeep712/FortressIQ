import { api, ApiError, apiVoid } from './client';

export interface AuthUser {
  user_id: string;
  email: string;
  full_name: string | null;
  avatar_url: string;
  is_verified: boolean;
  provider: 'email' | 'google';
  created_at: string;
}

export interface AuthResponse {
  user: AuthUser;
}

interface DeviceMetadata {
  device_name?: string | null;
  platform?: string | null;
  browser?: string | null;
}

export interface SessionRow {
  session_id: string;
  device_name: string | null;
  platform: string | null;
  browser: string | null;
  ip_address: string | null;
  user_agent: string | null;
  created_at: string;
  last_used_at: string | null;
  expires_at: string;
}

const device: DeviceMetadata = {
  device_name: undefined,
  platform: typeof navigator !== 'undefined' ? navigator.platform || null : null,
  browser: undefined,
};

const browserName = () => {
  if (typeof navigator === 'undefined') return null;
  const ua = navigator.userAgent;
  if (/Edg\//.test(ua)) return 'Edge';
  if (/OPR\//.test(ua)) return 'Opera';
  if (/Chrome\//.test(ua)) return 'Chrome';
  if (/Firefox\//.test(ua)) return 'Firefox';
  if (/Safari\//.test(ua)) return 'Safari';
  return null;
};

const registerBody = (): DeviceMetadata => ({
  ...device,
  platform: typeof navigator !== 'undefined' ? navigator.platform || null : null,
  browser: browserName(),
});

export function register(email: string, password: string, fullName: string): Promise<AuthResponse> {
  return api<AuthResponse>('/auth/register', {
    method: 'POST',
    body: { email, password, full_name: fullName, ...registerBody() },
  });
}

export function login(email: string, password: string): Promise<AuthResponse> {
  return api<AuthResponse>('/auth/login', {
    method: 'POST',
    body: { email, password, ...registerBody() },
  });
}

export function me(): Promise<AuthUser> {
  return api<AuthUser>('/auth/me');
}

export function logout(): Promise<void> {
  return apiVoid('/auth/logout', { method: 'POST' });
}

export function logoutAll(): Promise<void> {
  return apiVoid('/auth/logout-all', { method: 'POST' });
}

export function sessions(): Promise<SessionRow[]> {
  return api<SessionRow[]>('/auth/sessions');
}

export function forgotPassword(email: string): Promise<{ sent: boolean }> {
  return api<{ sent: boolean }>('/auth/forgot-password', {
    method: 'POST',
    body: { email },
  });
}

export interface VerifiedResponse {
  verified: boolean;
  user: AuthUser;
}

export function verifyEmail(token: string): Promise<VerifiedResponse> {
  return api<VerifiedResponse>('/auth/verify-email', {
    method: 'POST',
    body: { token },
  });
}

export function resendVerification(email: string): Promise<{ sent: boolean }> {
  return api<{ sent: boolean }>('/auth/resend-verification', {
    method: 'POST',
    body: { email },
  });
}

export function resetPassword(token: string, password: string): Promise<{ sent: boolean }> {
  return api<{ sent: boolean }>('/auth/reset-password', {
    method: 'POST',
    body: { token, password },
  });
}

// Point the browser at the backend's server-side Google OAuth redirect.
export function googleOAuthUrl(): string {
  return '/api/v1/auth/google';
}

export function mergeGoogleAccount(mergeToken: string, password: string): Promise<AuthResponse> {
  return api<AuthResponse>('/auth/google/merge', {
    method: 'POST',
    body: { merge_token: mergeToken, password, ...registerBody() },
  });
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError && typeof err.detail === 'string') {
    return err.detail;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return 'Something went wrong. Please try again.';
}
