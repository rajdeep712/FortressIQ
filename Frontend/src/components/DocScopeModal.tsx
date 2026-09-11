import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { X, FileText, Library, CheckSquare, Square } from 'lucide-react';
import { useViewerStore } from '../store/useViewerStore';
import { saveDocSelection } from '../utils/docSelection';

interface DocScopeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DocScopeModal: React.FC<DocScopeModalProps> = ({ isOpen, onClose }) => {
  const documents = useViewerStore((s) => s.documents);
  const setSelectedDocIds = useViewerStore((s) => s.setSelectedDocIds);
  const user = useViewerStore((s) => s.user);

  // `null` draft means "all documents checked".
  const [draft, setDraft] = useState<string[] | null>(null);

  useEffect(() => {
    if (isOpen) {
      const current = useViewerStore.getState().selectedDocIds;
      setDraft(current ?? documents.map((d) => d.id));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  if (!isOpen) return null;

  const allIds = documents.map((d) => d.id);
  const draftIds = draft === null ? allIds : draft;
  const allSelected = documents.length > 0 && draftIds.length === documents.length;

  const toggleDoc = (id: string) => {
    setDraft((prev) => {
      const ids = prev === null ? allIds : prev;
      return ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id];
    });
  };

  const apply = () => {
    // "All selected" collapses to null (default) so any newly uploaded
    // document joins the scope automatically.
    const commit = documents.length > 0 && draftIds.length === documents.length ? null : draftIds;
    setSelectedDocIds(commit);
    if (user?.user_id) saveDocSelection(user.user_id, commit);
    onClose();
  };

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

        <div className="flex items-center gap-3 mb-1">
          <div className="w-10 h-10 rounded-2xl bg-[#F97316]/15 text-[#EA580C] flex items-center justify-center">
            <Library className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-display text-xl font-bold text-[#221C16]">
              Documents for this query
            </h3>
            <p className="text-xs text-[#7A6D5E]">
              Only the selected papers are used to answer your questions.
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between mt-4 mb-2">
          <button
            type="button"
            onClick={() => setDraft(allSelected ? [] : null)}
            className="flex items-center gap-2 text-xs font-semibold text-[#C2410C] hover:text-[#EA580C] transition-colors cursor-pointer"
          >
            {allSelected ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4" />}
            {allSelected ? 'Deselect all' : 'Select all documents'}
          </button>
          <span className="text-[11px] font-mono text-[#8C7D6C] tabular-nums">
            {draftIds.length} / {documents.length} selected
          </span>
        </div>

        {documents.length === 0 ? (
          <div className="py-10 text-center text-xs text-[#8C7D6C]">
            No documents yet — upload one to start scoping your questions.
          </div>
        ) : (
          <div className="max-h-72 overflow-y-auto space-y-1.5 pr-1">
            {documents.map((doc) => {
              const checked = draftIds.includes(doc.id);
              return (
                <button
                  key={doc.id}
                  type="button"
                  onClick={() => toggleDoc(doc.id)}
                  className={`w-full text-left p-2.5 rounded-xl text-xs transition-all flex items-start gap-2.5 cursor-pointer border ${
                    checked
                      ? 'bg-[#F97316]/10 border-[#F97316]/30'
                      : 'bg-white/60 border-[#E5DEC3] hover:border-[#F97316]/40'
                  }`}
                >
                  <span className="mt-0.5 shrink-0">
                    {checked ? (
                      <CheckSquare className="w-4 h-4 text-[#EA580C]" />
                    ) : (
                      <Square className="w-4 h-4 text-[#B3A698]" />
                    )}
                  </span>
                  <FileText className="w-4 h-4 mt-0.5 shrink-0 text-[#8F8171]" />
                  <span className="flex-1 min-w-0">
                    <span className="block font-semibold text-[#221B14] line-clamp-1">
                      {doc.title}
                    </span>
                    <span className="block text-[11px] text-[#7F7060] mt-0.5">
                      {doc.filename} • {doc.totalPages} pages • {doc.fileSize}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        )}

        <div className="mt-4 flex items-center gap-2">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 py-2.5 rounded-xl bg-[#EFE9DF] text-[#3C3227] font-semibold text-sm hover:bg-[#E7E0D3] transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={apply}
            className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white font-semibold text-sm shadow-md shadow-[#F97316]/30 hover:opacity-95 transition-all cursor-pointer"
          >
            Apply
          </button>
        </div>

        {allSelected && (
          <p className="mt-3 text-[11px] text-[#8C7D6C] text-center">
            All documents selected — future uploads are included automatically.
          </p>
        )}
      </motion.div>
    </div>
  );
};