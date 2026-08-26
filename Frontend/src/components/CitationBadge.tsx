import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { FileText, ArrowUpRight, CheckCircle2 } from 'lucide-react';
import { Citation } from '../types';
import { useViewerStore } from '../store/useViewerStore';

interface CitationBadgeProps {
  citation: Citation;
}

export const CitationBadge: React.FC<CitationBadgeProps> = ({ citation }) => {
  const [isHovered, setIsHovered] = useState(false);
  const activeCitationId = useViewerStore((s) => s.activeCitationId);
  const jumpToCitation = useViewerStore((s) => s.jumpToCitation);

  const isActive = activeCitationId === citation.id;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    jumpToCitation(citation);
  };

  return (
    <span className="relative inline-block align-baseline mx-0.5" id={`citation-badge-wrapper-${citation.id}`}>
      <motion.button
        id={`citation-badge-${citation.id}`}
        type="button"
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.94 }}
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        aria-label={`View citation ${citation.id} from ${citation.docTitle} on page ${citation.page}`}
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-mono font-bold rounded-md transition-all duration-200 cursor-pointer select-none ${
          isActive
            ? 'bg-[#F97316] text-white ring-2 ring-[#F97316]/50 shadow-sm shadow-[#F97316]/30'
            : 'bg-[#F97316]/15 text-[#EA580C] hover:bg-[#F97316] hover:text-white border border-[#F97316]/30'
        }`}
      >
        <span>[{citation.id}]</span>
        <ArrowUpRight className="w-2.5 h-2.5 opacity-75" />
      </motion.button>

      {/* Floating Hover Preview Card */}
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 bg-[#FAF7F2] text-[#221C16] rounded-xl border border-[#E4DCCE] shadow-xl z-50 pointer-events-none text-left"
          >
            {/* Header info */}
            <div className="flex items-start justify-between gap-2 pb-2 border-b border-[#EBE3D5]">
              <div className="flex items-center gap-1.5">
                <span className="w-5 h-5 rounded-md bg-[#F97316] text-white flex items-center justify-center text-[10px] font-mono font-bold shrink-0">
                  {citation.id}
                </span>
                <span className="text-xs font-semibold text-[#3D3228] line-clamp-1">
                  {citation.chunkTitle || 'Retrieved Vector Chunk'}
                </span>
              </div>
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-[#FEF3C7] text-[#92400E] text-[10px] font-medium rounded-full shrink-0">
                <CheckCircle2 className="w-2.5 h-2.5 text-[#D97706]" />
                {Math.round(citation.confidence * 100)}% Match
              </span>
            </div>

            {/* Document and page info */}
            <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-[#786B5D]">
              <FileText className="w-3 h-3 text-[#A89A8B]" />
              <span className="truncate max-w-[150px] font-medium">{citation.docTitle}</span>
              <span>•</span>
              <span className="font-semibold text-[#EA580C]">Page {citation.page}</span>
            </div>

            {/* Snippet preview */}
            <div className="mt-2 text-xs text-[#4A3E31] bg-white/80 p-2 rounded-lg border border-[#E9E1D4] leading-relaxed italic line-clamp-3">
              "{citation.snippet}"
            </div>

            <div className="mt-2 pt-1.5 border-t border-[#EBE3D5] text-[10px] text-[#A89A8B] flex items-center justify-between">
              <span>Click to inspect & highlight in PDF viewer</span>
              <span className="font-mono text-[#F97316]">Top {citation.boundingBox.top}%</span>
            </div>

            {/* Tooltip triangle */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px w-2 h-2 border-r border-b border-[#E4DCCE] bg-[#FAF7F2] rotate-45" />
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
};
