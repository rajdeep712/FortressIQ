import React from 'react';

interface PageShellProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}

/**
 * Full-screen wrapper shared by all standalone auth pages (/verify,
 * /forgot-password, /reset-password). Matches the app theme: cream
 * background, serif display headline, orange "M" badge, soft orange glow.
 */
export const PageShell: React.FC<PageShellProps> = ({ title, subtitle, children }) => {
  return (
    <div className="min-h-screen w-full bg-[#F6F2EC] text-[#221C16] flex items-center justify-center p-4 select-none font-sans">
      <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden>
        <div className="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-[#FF7A00]/15 blur-3xl" />
        <div className="absolute -bottom-40 -right-24 w-[28rem] h-[28rem] rounded-full bg-[#EA580C]/15 blur-3xl" />
      </div>

      <div className="relative w-full max-w-md bg-[#FAF7F2] rounded-3xl border border-[#E5DEC3] shadow-2xl p-6 sm:p-8">
        <div className="text-center mb-6">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-[#FF7A00] to-[#E65100] flex items-center justify-center mx-auto mb-3 shadow-md shadow-[#F97316]/25 text-white font-bold text-lg font-mono">
            <span className="leading-none tracking-tighter">M</span>
          </div>
          <h3 className="font-display text-2xl font-bold text-[#221C16]">{title}</h3>
          {subtitle && <p className="text-xs text-[#7A6D5E] mt-1.5">{subtitle}</p>}
        </div>
        {children}
      </div>
    </div>
  );
};