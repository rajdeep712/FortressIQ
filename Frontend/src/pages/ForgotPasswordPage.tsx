import React, { useState } from 'react';
import {
  Mail,
  Loader2,
  ArrowLeft,
  AlertCircle,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import { errorMessage, forgotPassword } from '../api/auth';
import { PageShell } from './PageShell';

const goHome = () => {
  window.location.assign('/');
};

export const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      // Anti-enumeration: the backend always returns {sent: true}; we show
      // the same neutral confirmation for every email.
      await forgotPassword(email.trim());
      setSent(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell
      title="Reset your password"
      subtitle="Enter your email and we’ll send you a reset link."
    >
      {sent ? (
        <div className="space-y-5 text-center py-4">
          <div className="w-16 h-16 rounded-full bg-emerald-100 border border-emerald-200 flex items-center justify-center mx-auto">
            <CheckCircle2 className="w-9 h-9 text-emerald-600" />
          </div>
          <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-3 text-sm text-emerald-700">
            If that email has an account, a reset link is on its way. Check your inbox (and spam
            folder).
          </div>
          <button
            type="button"
            onClick={goHome}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer flex items-center justify-center gap-2"
          >
            Back to Sign In
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
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
            Send Reset Link
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