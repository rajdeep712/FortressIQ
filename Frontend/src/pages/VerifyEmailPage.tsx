import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import {
  CheckCircle2,
  AlertCircle,
  Loader2,
  ArrowRight,
  Mail,
  ArrowLeft,
} from 'lucide-react';
import { errorMessage, resendVerification, verifyEmail } from '../api/auth';
import { PageShell } from './PageShell';

type VerifyState = 'loading' | 'success' | 'error';

const goHome = () => {
  window.location.assign('/');
};

export const VerifyEmailPage: React.FC = () => {
  const [token] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('token') || '';
  });
  const [state, setState] = useState<VerifyState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(3);
  const [resendEmail, setResendEmail] = useState('');
  const [resendSent, setResendSent] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendError, setResendError] = useState<string | null>(null);
  const ranRef = useRef(false);

  useEffect(() => {
    // Fail loudly if the link is missing its token rather than silently
    // showing the error UI (also helps when the email client strips the query).
    if (ranRef.current) return;
    ranRef.current = true;

    if (!token) {
      setState('error');
      setError('This verification link is missing its token. It may be incomplete — please check the email link or request a new one.');
      return;
    }

    verifyEmail(token)
      .then(() => {
        setState('success');
      })
      .catch((err) => {
        setState('error');
        setError(
          errorMessage(err) ||
            'We could not verify this email. The link may be invalid or expired.'
        );
      });
  }, [token]);

  // 3-2-1 countdown then send the user back to the home page where the
  // session (access/refresh cookies) is still valid and they can sign in.
  useEffect(() => {
    if (state !== 'success') return;
    if (countdown <= 0) {
      goHome();
      return;
    }
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [state, countdown]);

  const handleResend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resendEmail.trim()) return;
    setResendLoading(true);
    setResendError(null);
    try {
      await resendVerification(resendEmail.trim());
      setResendSent(true);
    } catch (err) {
      setResendError(errorMessage(err));
    } finally {
      setResendLoading(false);
    }
  };

  return (
    <PageShell
      title={
        state === 'loading'
          ? 'Verifying your email'
          : state === 'success'
            ? 'Email verified'
            : 'Verification failed'
      }
      subtitle={
        state === 'loading'
          ? 'Just a moment while we confirm your address…'
          : state === 'success'
            ? 'Your account is ready to upload documents.'
            : 'We had trouble confirming this link.'
      }
    >
      {state === 'loading' && (
        <div className="py-8 text-center space-y-4">
          <Loader2 className="w-10 h-10 text-[#F97316] animate-spin mx-auto" />
          <p className="text-sm text-[#7A6D5E]">Checking the verification token…</p>
        </div>
      )}

      {state === 'success' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center py-4 space-y-5"
        >
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 320, damping: 18 }}
            className="w-16 h-16 rounded-full bg-emerald-100 border border-emerald-200 flex items-center justify-center mx-auto"
          >
            <CheckCircle2 className="w-9 h-9 text-emerald-600" />
          </motion.div>

          <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-3 text-sm text-emerald-700">
            Your email has been verified successfully. You can now sign in and upload documents.
          </div>

          <div className="text-xs text-[#8C7D6C]">
            Redirecting you to the home page to{' '}
            <span className="font-semibold text-[#EA580C]">sign in</span> in{' '}
            <span className="inline-flex items-center justify-center min-w-5 h-5 px-1 rounded-md bg-[#EFE9DF] font-mono font-bold text-[#221C16]">
              {countdown}
            </span>
            …
          </div>

          <div className="w-full h-1.5 bg-[#EFE9DF] rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-[#FF7A00] to-[#EA580C]"
              initial={{ width: '0%' }}
              animate={{ width: `${((3 - countdown) / 3) * 100}%` }}
              transition={{ duration: 0.9, ease: 'linear' }}
            />
          </div>

          <button
            type="button"
            onClick={goHome}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer flex items-center justify-center gap-2"
          >
            Go to Sign In
            <ArrowRight className="w-4 h-4" />
          </button>
        </motion.div>
      )}

      {state === 'error' && (
        <div className="space-y-4">
          <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>

          {/* Resend verification */}
          {!resendSent ? (
            <form onSubmit={handleResend} className="space-y-3">
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
                <input
                  type="email"
                  required
                  value={resendEmail}
                  onChange={(e) => setResendEmail(e.target.value)}
                  placeholder="Email address"
                  className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
                />
              </div>
              {resendError && (
                <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-700">
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>{resendError}</span>
                </div>
              )}
              <button
                type="submit"
                disabled={resendLoading}
                className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {resendLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                Resend verification link
              </button>
            </form>
          ) : (
            <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-3 py-2.5 text-xs text-emerald-700">
              If that email has an account, a fresh verification link is on its way.
            </div>
          )}

          <button
            type="button"
            onClick={goHome}
            className="w-full py-2.5 rounded-xl bg-[#EFE9DF] text-[#3C3227] font-semibold text-sm hover:bg-[#E7E0D3] transition-colors cursor-pointer flex items-center justify-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to home
          </button>
        </div>
      )}
    </PageShell>
  );
};