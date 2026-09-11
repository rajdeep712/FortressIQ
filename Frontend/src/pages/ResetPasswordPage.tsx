import React, { useEffect, useState } from 'react';
import {
  Lock,
  Loader2,
  ArrowLeft,
  AlertCircle,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import { errorMessage, resetPassword } from '../api/auth';
import { PageShell } from './PageShell';

const goHome = () => {
  window.location.assign('/');
};

export const ResetPasswordPage: React.FC = () => {
  const [token] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('token') || '';
  });
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [countdown, setCountdown] = useState(3);

  useEffect(() => {
    if (!done) return;
    if (countdown <= 0) {
      goHome();
      return;
    }
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [done, countdown]);

  const missingToken = !token;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading || missingToken) return;
    setLoading(true);
    setError(null);
    try {
      await resetPassword(token, password);
      setDone(true);
    } catch (err) {
      setError(errorMessage(err) || 'We could not reset your password. The link may have expired.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell
      title={done ? 'Password updated' : 'Set a new password'}
      subtitle={
        missingToken
          ? 'This reset link is missing its token.'
          : done
            ? 'You can now sign in with your new password.'
            : 'Choose a new password for your account.'
      }
    >
      {done ? (
        <div className="space-y-5 text-center py-4">
          <div className="w-16 h-16 rounded-full bg-emerald-100 border border-emerald-200 flex items-center justify-center mx-auto">
            <CheckCircle2 className="w-9 h-9 text-emerald-600" />
          </div>
          <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-3 text-sm text-emerald-700">
            Your password has been updated successfully.
          </div>
          <div className="text-xs text-[#8C7D6C]">
            Redirecting you to sign in in{' '}
            <span className="inline-flex items-center justify-center min-w-5 h-5 px-1 rounded-md bg-[#EFE9DF] font-mono font-bold text-[#221C16]">
              {countdown}
            </span>
            …
          </div>
          <button
            type="button"
            onClick={goHome}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer flex items-center justify-center gap-2"
          >
            Go to Sign In
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      ) : missingToken ? (
        <div className="space-y-4">
          <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>
              The reset link is incomplete — please open the link exactly as it arrived in your
              email, or request a new one.
            </span>
          </div>
          <button
            type="button"
            onClick={goHome}
            className="w-full py-2.5 rounded-xl bg-[#EFE9DF] text-[#3C3227] font-semibold text-sm hover:bg-[#E7E0D3] transition-colors cursor-pointer flex items-center justify-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Sign In
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="New password (min 8 chars)"
              className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
            />
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            Update Password
            {!loading && <ArrowRight className="w-4 h-4" />}
          </button>

          <button
            type="button"
            onClick={goHome}
            className="w-full py-2.5 rounded-xl bg-[#EFE9DF] text-[#3C3227] font-semibold text-sm hover:bg-[#E7E0D3] transition-colors cursor-pointer flex items-center justify-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Sign In
          </button>
        </form>
      )}
    </PageShell>
  );
};