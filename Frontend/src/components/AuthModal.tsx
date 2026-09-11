import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { X, Mail, Lock, User, Loader2, ArrowRight, AlertCircle } from 'lucide-react';
import {
  errorMessage,
  forgotPassword,
  googleOAuthUrl,
  login,
  mergeGoogleAccount,
  register,
  resetPassword,
} from '../api/auth';
import { useViewerStore } from '../store/useViewerStore';

export type AuthMode = 'signin' | 'signup' | 'forgot' | 'reset' | 'merge';

export interface ResolveMergeInfo {
  token: string;
  email: string;
  name?: string;
  avatar?: string;
}

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialMode?: AuthMode;
  resetToken?: string;
  merge?: ResolveMergeInfo | null;
}

export const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  initialMode = 'signin',
  resetToken = '',
  merge = null,
}) => {
  const setUser = useViewerStore((s) => s.setUser);
  const setShowAuthModal = useViewerStore((s) => s.setShowAuthModal);

  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [resetNewPassword, setResetNewPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // Re-sync internal state whenever the modal is opened with a specific mode.
  useEffect(() => {
    if (isOpen) {
      setMode(merge ? 'merge' : initialMode);
      setError(null);
      setInfo(null);
      setPassword('');
      setResetNewPassword('');
      if (merge) setEmail(merge.email);
    }
  }, [isOpen, initialMode, merge]);

  if (!isOpen) return null;

  const done = async () => {
    setShowAuthModal(false);
    onClose();
  };

  const handleGoogle = () => {
    window.location.href = googleOAuthUrl();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setInfo(null);

    if (mode === 'forgot') {
      setLoading(true);
      try {
        await forgotPassword(email);
        setInfo('If that email has an account, a reset link is on its way.');
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setLoading(false);
      }
      return;
    }

    if (mode === 'reset') {
      setLoading(true);
      try {
        await resetPassword(resetToken, resetNewPassword);
        setInfo('Password updated. You can now sign in with your new password.');
        setMode('signin');
        setPassword('');
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setLoading(false);
      }
      return;
    }

    if (mode === 'merge') {
      if (!merge) return;
      setLoading(true);
      try {
        const res = await mergeGoogleAccount(merge.token, password);
        setUser(res.user);
        await done();
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setLoading(false);
      }
      return;
    }

    setLoading(true);
    try {
      const res =
        mode === 'signup'
          ? await register(email, password, name)
          : await login(email, password);
      setUser(res.user);
      if (mode === 'signup') {
        // Banner nudges the user to verify before uploads are unlocked, then we
        // automatically return to the home page (session cookies are already set).
        setInfo(`Account created! Check ${email} for a verification link to enable uploads.`);
        window.setTimeout(() => void done(), 2500);
      } else {
        await done();
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const title =
    mode === 'signin'
      ? 'Welcome back'
      : mode === 'signup'
        ? 'Create your account'
        : mode === 'forgot'
          ? 'Reset your password'
          : mode === 'merge'
            ? 'Link your Google account'
            : 'Set a new password';

  const subtitle =
    mode === 'signin'
      ? 'Sign in to upload documents and start asking questions.'
      : mode === 'signup'
        ? 'Get started for free — no credit card required.'
        : mode === 'forgot'
          ? 'Enter your email and we’ll send you a reset link.'
          : mode === 'merge'
            ? `An account already exists for ${merge?.email ?? 'this email'}. Enter its password to link your Google login.`
            : 'Choose a new password for your account.';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
      <motion.div
        initial={{ opacity: 0, scale: 0.94, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.94, y: 10 }}
        className="w-full max-w-md bg-[#FAF7F2] rounded-3xl border border-[#E5DEC3] shadow-2xl p-6 relative overflow-hidden"
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 rounded-full hover:bg-[#EFE9DF] text-[#7A6E60] transition-colors cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="text-center mb-5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-[#FF7A00] to-[#E65100] flex items-center justify-center mx-auto mb-3 shadow-md shadow-[#F97316]/25 text-white font-bold text-lg font-mono">
            <span className="leading-none tracking-tighter">M</span>
          </div>
          <h3 className="font-display text-xl font-bold text-[#221C16]">{title}</h3>
          <p className="text-xs text-[#7A6D5E] mt-1">{subtitle}</p>
        </div>

        {/* Tabs (only for the two primary modes) */}
        {(mode === 'signin' || mode === 'signup') && (
          <div className="flex bg-[#EFE9DE] rounded-xl p-0.5 mb-4">
            {(['signin', 'signup'] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => {
                  setMode(tab);
                  setError(null);
                  setInfo(null);
                }}
                className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                  mode === tab
                    ? 'bg-white text-[#221C16] shadow-sm'
                    : 'text-[#8C7D6C] hover:text-[#221C16]'
                }`}
              >
                {tab === 'signin' ? 'Sign In' : 'Create Account'}
              </button>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-3">
          {mode === 'signup' && (
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Full name"
                className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
              />
            </div>
          )}

          {mode !== 'reset' && mode !== 'merge' && (
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email address"
                className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
              />
            </div>
          )}

          {/* Merge shows the matched email non-editable */}
          {mode === 'merge' && (
            <div className="rounded-xl bg-[#EFE9DE] border border-[#E5DEC3] px-3 py-2.5 text-sm text-[#5C5042] flex items-center gap-2">
              <Mail className="w-4 h-4 text-[#A89A8B]" />
              <span className="truncate">{merge?.email}</span>
            </div>
          )}

          {(mode === 'signin' || mode === 'signup' || mode === 'merge') && (
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                minLength={mode === 'signup' ? 8 : undefined}
                placeholder={mode === 'merge' ? 'Enter your password' : 'Password'}
                className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
              />
            </div>
          )}

          {mode === 'reset' && (
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
              <input
                type="password"
                required
                minLength={8}
                value={resetNewPassword}
                onChange={(e) => setResetNewPassword(e.target.value)}
                placeholder="New password (min 8 chars)"
                className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
              />
            </div>
          )}

          {mode === 'signin' && (
            <div className="text-right">
              <button
                type="button"
                onClick={() => {
                  onClose();
                  window.location.assign('/forgot-password');
                }}
                className="text-[11px] text-[#EA580C] hover:underline cursor-pointer"
              >
                Forgot password?
              </button>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {info && (
            <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-3 py-2.5 text-xs text-emerald-700">
              {info}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {mode === 'signin'
              ? 'Sign In'
              : mode === 'signup'
                ? 'Create Account'
                : mode === 'forgot'
                  ? 'Send Reset Link'
                  : mode === 'merge'
                    ? 'Link Accounts'
                    : 'Update Password'}
            {!loading && <ArrowRight className="w-4 h-4" />}
          </button>
        </form>

        {/* Google / secondary actions */}
        {(mode === 'signin' || mode === 'signup') && (
          <>
            <div className="flex items-center gap-3 my-4">
              <div className="flex-1 h-px bg-[#E5DEC3]" />
              <span className="text-[11px] text-[#A89A8B] font-medium">or</span>
              <div className="flex-1 h-px bg-[#E5DEC3]" />
            </div>

            <button
              type="button"
              onClick={handleGoogle}
              className="w-full py-2.5 rounded-xl bg-white border border-[#E5DEC3] text-[#3C3227] font-semibold text-sm flex items-center justify-center gap-2 hover:bg-[#F5EFE4] transition-colors cursor-pointer"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4" />
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
              </svg>
              Continue with Google
            </button>
          </>
        )}

        {mode === 'forgot' && (
          <p className="text-center text-[11px] text-[#A89A8B] mt-4">
            <button
              type="button"
              onClick={() => setMode('signin')}
              className="text-[#EA580C] hover:underline cursor-pointer"
            >
              Back to sign in
            </button>
          </p>
        )}
      </motion.div>
    </div>
  );
};
