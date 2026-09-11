import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import {
  Upload,
  X,
  FileText,
  Loader2,
  AlertCircle,
  RefreshCw,
  Lock,
  ShieldAlert,
  Mail,
  CheckCircle2,
  LogIn,
} from 'lucide-react';
import { useViewerStore } from '../store/useViewerStore';
import { RAGDocument } from '../types';
import {
  retryDocument,
  uploadDocument,
  watchDocumentStatus,
  DOC_FAILED_STATUS,
  DocumentStatusResponse,
} from '../api/documents';
import { errorMessage, resendVerification } from '../api/auth';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type Phase = 'idle' | 'working' | 'failed' | 'completed';

interface Stage {
  active: string;
  done: string;
}

const STAGES: Stage[] = [
  { active: 'Uploading to storage…', done: 'Uploaded to storage' },
  { active: 'Parsing text, tables & layout…', done: 'Parsed' },
  { active: 'Splitting into overlapping chunks…', done: 'Chunked' },
  { active: 'Generating embeddings…', done: 'Indexed' },
  { active: 'Upserting vectors to Qdrant…', done: 'Completed' },
];

const LAST_STAGE = STAGES.length;

// Number of stages the backend has confirmed complete for a given status
// (UPLOADED/PROCESSING confirm only the upload step; the rest are emitted
// by the ingestion worker via the SSE event stream).
const CONFIRMED_BY_STATUS: Record<string, number> = {
  UPLOADED: 1,
  PROCESSING: 1,
  PARSED: 2,
  CHUNKED: 3,
  EMBEDDED: 4,
  COMPLETED: 5,
};

const TICK_MS = 700;

const getFileType = (name: string): RAGDocument['fileType'] => {
  const ext = name.split('.').pop()?.toLowerCase();
  if (ext === 'pdf') return 'pdf';
  if (ext === 'doc' || ext === 'docx' || ext === 'odt') return 'doc';
  return 'txt';
};

const formatBytes = (bytes: number): string => {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
};

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose }) => {
  const user = useViewerStore((s) => s.user);
  const isAuthenticated = useViewerStore((s) => s.isAuthenticated);
  const uploadDocumentToStore = useViewerStore((s) => s.uploadDocument);
  const setShowAuthModal = useViewerStore((s) => s.setShowAuthModal);

  const [dragActive, setDragActive] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [fileName, setFileName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(0);
  const [shown, setShown] = useState(0);
  const [lastStatus, setLastStatus] = useState<DocumentStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [docId, setDocId] = useState<string | null>(null);
  const watchRef = useRef<{ close: () => void } | null>(null);
  const confirmedRef = useRef(0);

  // Verify-your-email panel state
  const [resendEmail, setResendEmail] = useState('');
  const [resendSent, setResendSent] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendError, setResendError] = useState<string | null>(null);

  const closeWatch = () => {
    watchRef.current?.close();
    watchRef.current = null;
  };

  useEffect(() => {
    confirmedRef.current = confirmed;
  }, [confirmed]);

  // Tear down the SSE stream when the modal unmounts.
  useEffect(() => closeWatch, []);

  // Advance the animated stepper one stage per tick, capped at the most
  // recent status confirmed by the backend.
  useEffect(() => {
    if (phase !== 'working') return;
    const id = window.setInterval(() => {
      setShown((prev) => (prev < confirmedRef.current ? prev + 1 : prev));
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [phase]);

  // Once every stage is confirmed and shown, move to the completed panel.
  useEffect(() => {
    if (phase !== 'working') return;
    if (shown < LAST_STAGE || confirmed < LAST_STAGE) return;
    setPhase('completed');
    if (lastStatus) uploadDocumentToStore(buildRagDocument(lastStatus));
  }, [phase, shown, confirmed, lastStatus]);

  const buildRagDocument = (s: DocumentStatusResponse): RAGDocument => ({
    id: s.doc_id,
    title: s.filename.replace(/\.[^/.]+$/, ''),
    filename: s.filename,
    // Page viewer is implemented later; real uploads show as ingested and
    // awaiting retrieval with no consumer-facing pages yet.
    totalPages: s.chunk_count || 0,
    fileType: getFileType(s.filename),
    uploadedAt: 'Just now',
    fileSize: file ? formatBytes(file.size) : '—',
    category: 'Custom Upload',
    authors: user?.full_name || undefined,
    abstract: 'Real uploaded document, indexed and available for retrieval.',
    pages: [],
    chunks: [],
  });

  const startWatch = (id: string) => {
    closeWatch();
    watchRef.current = watchDocumentStatus(id, {
      onStatus: (s) => {
        setLastStatus(s);
        const idx = CONFIRMED_BY_STATUS[s.status];
        if (idx !== undefined) setConfirmed(idx);
        if (s.status === DOC_FAILED_STATUS) {
          closeWatch();
          setPhase('failed');
          setError('Server-side ingestion failed for this document.');
        }
      },
    });
  };

  const reset = () => {
    closeWatch();
    setPhase('idle');
    setFileName('');
    setFile(null);
    setConfirmed(0);
    setShown(0);
    setLastStatus(null);
    setError(null);
    setDocId(null);
  };

  const handleUpload = async (f: File) => {
    closeWatch();
    setFileName(f.name);
    setFile(f);
    setError(null);
    setPhase('working');
    setConfirmed(0);
    setShown(0);
    setLastStatus(null);
    try {
      const res = await uploadDocument(f);
      setDocId(res.doc_id);
      // Upload done — tick the first stage green and the parsing spinner on.
      setConfirmed(1);
      setShown(1);
      startWatch(res.doc_id);
    } catch (err) {
      setPhase('failed');
      setError(errorMessage(err));
    }
  };

  const handleRetry = async () => {
    if (!docId) return;
    setRetrying(true);
    setError(null);
    setPhase('working');
    setConfirmed(1);
    setShown(1);
    try {
      await retryDocument(docId);
      startWatch(docId);
    } catch (err) {
      setPhase('failed');
      setError(errorMessage(err));
    } finally {
      setRetrying(false);
    }
  };

  const handleResend = async (e: React.FormEvent) => {
    e.preventDefault();
    const email = resendEmail.trim() || user?.email || '';
    if (!email) return;
    setResendLoading(true);
    setResendError(null);
    try {
      await resendVerification(email);
      setResendSent(true);
    } catch (err) {
      setResendError(errorMessage(err));
    } finally {
      setResendLoading(false);
    }
  };

  if (!isOpen) return null;

  const close = () => {
    closeWatch();
    onClose();
  };

  /* ----- Gate: not signed in ----- */
  if (!isAuthenticated) {
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
            onClick={close}
            className="absolute top-5 right-5 p-1.5 rounded-full hover:bg-[#EFE9DF] text-[#7A6E60] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>

          <div className="w-14 h-14 rounded-2xl bg-[#F97316]/15 text-[#EA580C] flex items-center justify-center mx-auto mb-4">
            <Lock className="w-7 h-7" />
          </div>
          <h3 className="font-display text-xl font-bold text-[#221C16] text-center mb-2">
            Sign in to upload documents
          </h3>
          <p className="text-xs text-[#7A6D5E] text-center mb-6">
            Accounts let you store, index and chat with your research papers.
          </p>
          <button
            type="button"
            onClick={() => {
              close();
              setShowAuthModal(true);
            }}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer flex items-center justify-center gap-2"
          >
            <LogIn className="w-4 h-4" />
            Sign In / Create Account
          </button>
          <button
            type="button"
            onClick={close}
            className="w-full mt-2 py-2.5 rounded-xl bg-[#EFE9DF] text-[#3C3227] font-semibold text-sm hover:bg-[#E7E0D3] transition-colors cursor-pointer"
          >
            Not now
          </button>
        </motion.div>
      </div>
    );
  }

  /* ----- Gate: signed in but email not verified ----- */
  if (user && !user.is_verified) {
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
            onClick={close}
            className="absolute top-5 right-5 p-1.5 rounded-full hover:bg-[#EFE9DF] text-[#7A6E60] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>

          <div className="w-14 h-14 rounded-2xl bg-[#F97316]/15 text-[#EA580C] flex items-center justify-center mx-auto mb-4">
            <ShieldAlert className="w-7 h-7" />
          </div>
          <h3 className="font-display text-xl font-bold text-[#221C16] text-center mb-2">
            Verify your email first
          </h3>
          <p className="text-xs text-[#7A6D5E] text-center mb-6 leading-relaxed">
            We sent a verification link to <span className="font-semibold text-[#221C16]">{user.email}</span>.
            Uploads unlock once your email is confirmed.
          </p>

          {!resendSent ? (
            <form onSubmit={handleResend} className="space-y-3">
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#A89A8B]" />
                <input
                  type="email"
                  required
                  value={resendEmail || user.email}
                  onChange={(e) => setResendEmail(e.target.value)}
                  placeholder="Email address"
                  className="w-full bg-white border border-[#E5DEC3] rounded-xl pl-10 pr-3 py-2.5 text-sm text-[#221C16] placeholder-[#A89A8B] outline-none focus:border-[#F97316] transition-colors"
                />
              </div>
              {resendError && (
                <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700">
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
            <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-3 text-sm text-emerald-700 text-center flex items-center justify-center gap-2">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              Fresh verification link sent — check your inbox, then upload again.
            </div>
          )}

          <button
            type="button"
            onClick={close}
            className="w-full mt-3 py-2.5 rounded-xl bg-[#EFE9DF] text-[#3C3227] font-semibold text-sm hover:bg-[#E7E0D3] transition-colors cursor-pointer"
          >
            Close
          </button>
        </motion.div>
      </div>
    );
  }

  /* ----- Main flow ----- */
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
      <motion.div
        initial={{ opacity: 0, scale: 0.94, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.94, y: 10 }}
        className="w-full max-w-lg bg-[#FAF7F2] rounded-3xl border border-[#E5DEC3] shadow-2xl p-6 relative overflow-hidden"
      >
        <button
          type="button"
          onClick={close}
          className="absolute top-5 right-5 p-1.5 rounded-full hover:bg-[#EFE9DF] text-[#7A6E60] transition-colors cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-2xl bg-[#F97316]/15 text-[#EA580C] flex items-center justify-center">
            <Upload className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-display text-xl font-bold text-[#221C16]">
              Upload New Document
            </h3>
            <p className="text-xs text-[#7A6D5E]">
              Real file → ingest → index into the vector store
            </p>
          </div>
        </div>

        {phase === 'working' ? (
          <div className="py-4 space-y-3">
            <div className="flex items-center justify-between mb-1">
              <h4 className="text-sm font-bold text-[#221C16] truncate">
                {fileName ? `Indexing ${fileName}` : 'Indexing document…'}
              </h4>
              <span className="text-[11px] text-[#8C7D6C] tabular-nums">
                {Math.min(shown, LAST_STAGE)}/5
              </span>
            </div>

            {STAGES.map((stage, i) => {
              const done = i < shown;
              const active = i === shown;
              return (
                <div
                  key={stage.done}
                  className={`flex items-center gap-3 rounded-xl px-3 py-2.5 border transition-colors ${
                    done
                      ? 'bg-emerald-50/60 border-emerald-100'
                      : 'bg-white/50 border-[#E5DEC3]'
                  }`}
                >
                  {done ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                  ) : active ? (
                    <Loader2 className="w-5 h-5 text-[#F97316] animate-spin shrink-0" />
                  ) : (
                    <div className="w-5 h-5 rounded-full border-2 border-[#D8CDBB] shrink-0" />
                  )}
                  <span
                    className={`text-sm ${
                      done
                        ? 'text-emerald-800 font-semibold'
                        : active
                          ? 'text-[#221C16] font-semibold'
                          : 'text-[#8C7D6C]'
                    }`}
                  >
                    {done ? stage.done : stage.active}
                  </span>
                </div>
              );
            })}

            <p className="text-[11px] text-[#8C7D6C] pt-1">
              Background worker is streaming each stage live — you can track
              progress here until it completes.
            </p>
          </div>
        ) : phase === 'completed' ? (
          <div className="py-8 text-center space-y-4">
            <div className="w-14 h-14 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-[#221C16] mb-1">
                Document ready
              </h4>
              <p className="text-xs text-[#8C7D6C] leading-relaxed">
                {fileName ? (
                  <>
                    <span className="font-semibold text-[#221C16]">{fileName}</span>{' '}
                    has been uploaded, indexed and upserted to the vector store.
                  </>
                ) : (
                  'Your document has been uploaded, indexed and upserted to the vector store.'
                )}
              </p>
            </div>
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-2 py-2.5 px-5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer"
            >
              <Upload className="w-4 h-4" />
              Upload another file
            </button>
          </div>
        ) : phase === 'failed' ? (
          <div className="py-4 space-y-2">
            <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-3 py-2.5 text-xs text-red-700">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error ?? 'Something went wrong while processing this document.'}</span>
            </div>
            {docId && (
              <button
                type="button"
                onClick={handleRetry}
                disabled={retrying}
                className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {retrying ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
                Retry ingestion
              </button>
            )}
          </div>
        ) : (
          <div>
            {/* Drag and drop / file picker */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                const f = e.dataTransfer.files?.[0];
                if (f) void handleUpload(f);
              }}
              onClick={() => document.getElementById('upload-file-input')?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${
                dragActive
                  ? 'border-[#F97316] bg-[#F97316]/10'
                  : 'border-[#DDD2C0] hover:border-[#F97316]/60 bg-white/60 hover:bg-white'
              }`}
            >
              <FileText className="w-10 h-10 text-[#EA580C] mx-auto mb-3" />
              <p className="text-sm font-bold text-[#221C16] mb-1">
                Click or drag & drop your document here
              </p>
              <p className="text-xs text-[#8A7B6B]">
                Supports PDF, DOCX, TXT — max 50MB
              </p>
              <input
                id="upload-file-input"
                type="file"
                className="hidden"
                accept=".pdf,.doc,.docx,.txt,.md,.odt"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleUpload(f);
                  e.target.value = '';
                }}
              />
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};