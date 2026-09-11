import React, { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  FileText,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Search,
  BookOpen,
  Sparkles,
  Layers,
  Upload,
  ArrowRight,
  X,
  ChevronDown,
  Check,
  LogOut
} from 'lucide-react';
import { useViewerStore } from '../store/useViewerStore';
import { logout } from '../api/auth';
import type { AuthMode } from './AuthModal';
import { RAGDocument } from '../types';
import BlurText from './reactbits/BlurText';

interface DocumentViewerProps {
  onOpenUploadModal?: () => void;
  onOpenAuth?: (mode?: AuthMode) => void;
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({ onOpenUploadModal, onOpenAuth }) => {
  const {
    activeDocumentId,
    activePage,
    documents,
    activeCitationId,
    activeHighlight,
    pulseTrigger,
    zoomLevel,
    searchQueryInDoc,
    isOutlineOpen,
    setActiveDocumentId,
    setActivePage,
    zoomIn,
    zoomOut,
    resetZoom,
    setSearchQueryInDoc,
    setIsOutlineOpen,
    clearActiveCitation,
    isAuthenticated,
    isSessionRestoring,
    isLoadingDocuments,
    user,
    setUser,
    setShowAuthModal
  } = useViewerStore();

  const [isDocDropdownOpen, setIsDocDropdownOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [showSearchInput, setShowSearchInput] = useState(false);
  const highlightRef = useRef<HTMLDivElement>(null);
  const documentContainerRef = useRef<HTMLDivElement>(null);

  const handleLogout = async () => {
    try {
      await logout();
    } catch {
      // The backend may already have rotated/cleared the session; clear
      // local state regardless.
    }
    setUser(null);
    setIsUserMenuOpen(false);
  };

  const currentDoc = documents.find((d) => d.id === activeDocumentId) || null;
  const currentPageData = currentDoc?.pages.find((p) => p.pageNumber === activePage) || null;

  const FEATURES = [
    {
      icon: <Upload className="w-7 h-7" />,
      title: 'Upload any document',
      desc: 'Drag & drop any PDF, DOCX, or TXT file to instantly index it into the vector store.'
    },
    {
      icon: <FileText className="w-7 h-7" />,
      title: 'Grounded citations',
      desc: 'Every answer cites the exact passage it is drawn from — numbered badges you can click.'
    },
    {
      icon: <Sparkles className="w-7 h-7" />,
      title: 'Jump to the highlighter',
      desc: 'Click a citation to see the source chunk highlighted right in the document viewer.'
    },
    {
      icon: <Layers className="w-7 h-7" />,
      title: 'Multi-document search',
      desc: 'Ask questions across multiple indexed documents at once.'
    }
  ];

  // Auto-scroll highlight into view when pulseTrigger updates
  useEffect(() => {
    if (activeHighlight && highlightRef.current) {
      setTimeout(() => {
        highlightRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }, 150);
    }
  }, [pulseTrigger, activePage, activeDocumentId]);

  const handlePrevPage = () => {
    if (activePage > 1) {
      setActivePage(activePage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentDoc && activePage < currentDoc.totalPages) {
      setActivePage(activePage + 1);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#EDE6DC] relative overflow-hidden select-text" id="document-viewer-container">
      {/* Header Bar */}
      <header className="px-4 py-3 bg-[#FAF7F2] border-b border-[#E4DCCE] flex items-center justify-between gap-3 shrink-0 z-30 shadow-xs">
        {/* Document Selector Dropdown */}
        {isAuthenticated && !isSessionRestoring && (
          <div className="relative flex items-center gap-2 min-w-0">
          <div className="p-2 rounded-xl bg-[#F97316]/10 text-[#EA580C] shrink-0">
            <FileText className="w-4 h-4" />
          </div>

          {isLoadingDocuments ? (
            /* Skeleton for the documents selection line */
            <div className="min-w-0 space-y-1.5" aria-label="Loading documents">
              <div className="h-3.5 w-40 rounded bg-[#E6DCCB] animate-pulse" />
              <div className="h-2 w-28 rounded bg-[#EEE6D9] animate-pulse" />
            </div>
          ) : documents.length === 0 ? (
            /* Signed in, no documents yet */
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 text-left text-xs sm:text-sm font-bold text-[#8A7B6B]">
                <span className="truncate max-w-[180px] sm:max-w-[280px]">No Document Selected</span>
              </div>
              <div className="text-[11px] text-[#B3A698]">No documents yet</div>
            </div>
          ) : (
            <>
            <div className="min-w-0">
              <button
                id="doc-selector-dropdown-btn"
                type="button"
                onClick={() => setIsDocDropdownOpen(!isDocDropdownOpen)}
                className="flex items-center gap-1.5 text-left text-xs sm:text-sm font-bold text-[#271F17] hover:text-[#EA580C] transition-colors cursor-pointer group"
              >
                <span className="truncate max-w-[180px] sm:max-w-[280px]">
                  {currentDoc ? currentDoc.title : 'No Document Selected'}
                </span>
                <ChevronDown className="w-3.5 h-3.5 text-[#8C7D6C] group-hover:text-[#EA580C] shrink-0" />
              </button>
              {currentDoc && (
                <div className="text-[11px] text-[#8C7D6C] flex items-center gap-2 mt-0.5">
                  <span className="font-mono">{currentDoc.filename}</span>
                  <span>•</span>
                  <span>{currentDoc.fileSize}</span>
                  <span>•</span>
                  <span className="hidden sm:inline font-medium text-[#D95D0F]">{currentDoc.category}</span>
                </div>
              )}
            </div>

            {/* Document Picker Dropdown Menu */}
            <AnimatePresence>
              {isDocDropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: 6, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 4, scale: 0.95 }}
                transition={{ duration: 0.12 }}
                className="absolute top-full left-0 mt-2 w-80 sm:w-96 p-2 bg-[#FAF7F2] rounded-2xl border border-[#E5DEC3] shadow-2xl z-50 text-left"
              >
                <div className="px-3 py-2 text-[11px] font-bold text-[#8C7E6E] uppercase tracking-wider flex items-center justify-between border-b border-[#EDE6DC] pb-2 mb-1">
                  <span>Indexed Papers in Vector Store</span>
                  <span className="font-mono text-[#EA580C]">{documents.length} Docs</span>
                </div>

                <div className="max-h-72 overflow-y-auto space-y-1 py-1">
                  {documents.map((doc) => {
                    const isSelected = doc.id === activeDocumentId;
                    return (
                      <button
                        key={doc.id}
                        type="button"
                        onClick={() => {
                          setActiveDocumentId(doc.id, 1);
                          setIsDocDropdownOpen(false);
                        }}
                        className={`w-full text-left p-2.5 rounded-xl text-xs transition-all flex items-start gap-2.5 cursor-pointer ${
                          isSelected
                            ? 'bg-[#F97316]/15 border border-[#F97316]/40 text-[#241C15]'
                            : 'hover:bg-[#EFEAE1] text-[#3D3328] border border-transparent'
                        }`}
                      >
                        <FileText className={`w-4 h-4 mt-0.5 shrink-0 ${isSelected ? 'text-[#EA580C]' : 'text-[#8F8171]'}`} />
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold text-xs text-[#221B14] line-clamp-1">
                            {doc.title}
                          </div>
                          <div className="text-[11px] text-[#7F7060] flex items-center gap-2 mt-0.5">
                            <span>{doc.totalPages} pages</span>
                            <span>•</span>
                            <span className="font-mono">{doc.fileSize}</span>
                            <span>•</span>
                            <span className="truncate">{doc.category}</span>
                          </div>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-[#EA580C] shrink-0 mt-1" />}
                      </button>
                    );
                  })}
                </div>

                <div className="pt-2 border-t border-[#EDE6DC] mt-1">
                  <button
                    type="button"
                    onClick={() => {
                      setIsDocDropdownOpen(false);
                      onOpenUploadModal?.();
                    }}
                    className="w-full py-2 px-3 rounded-xl bg-[#F6F0E6] hover:bg-[#EFE9DE] text-[#C2410C] font-semibold text-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                  >
                    <span>+ Upload New PDF to Index</span>
                  </button>
                </div>
              </motion.div>
            )}
            </AnimatePresence>
            </>
          )}
          </div>
        )}

        {/* Center & Right Controls: Page Switcher, Search, Zoom */}
        {currentDoc && (
          <div className="flex items-center gap-2 shrink-0">
            {/* Page Navigation Pill */}
            <div className="flex items-center bg-[#EFE9DF] rounded-xl border border-[#E0D7C8] p-0.5 text-xs font-semibold text-[#3C3227]">
              <button
                id="doc-prev-page-btn"
                type="button"
                onClick={handlePrevPage}
                disabled={activePage <= 1}
                className="p-1.5 hover:bg-[#FAF7F2] disabled:opacity-35 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
                title="Previous Page"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>

              <div className="px-2.5 py-1 text-xs font-mono">
                <span className="font-bold text-[#EA580C]">{activePage}</span>
                <span className="text-[#8C7D6C] mx-1">/</span>
                <span className="text-[#5F5142]">{currentDoc.totalPages}</span>
              </div>

              <button
                id="doc-next-page-btn"
                type="button"
                onClick={handleNextPage}
                disabled={activePage >= currentDoc.totalPages}
                className="p-1.5 hover:bg-[#FAF7F2] disabled:opacity-35 rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
                title="Next Page"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {/* Outline / Thumbnails toggle */}
            <button
              id="doc-outline-toggle-btn"
              type="button"
              onClick={() => setIsOutlineOpen(!isOutlineOpen)}
              title="Toggle Page Outline"
              className={`p-2 rounded-xl border transition-colors cursor-pointer ${
                isOutlineOpen
                  ? 'bg-[#F97316] text-white border-[#EA580C]'
                  : 'bg-[#EFE9DF] text-[#6C5E4E] hover:bg-[#E7E0D3] border-[#E0D7C8]'
              }`}
            >
              <Layers className="w-4 h-4" />
            </button>

            {/* Search in Doc Toggle */}
            <button
              id="doc-search-toggle-btn"
              type="button"
              onClick={() => setShowSearchInput(!showSearchInput)}
              title="Search within document"
              className={`p-2 rounded-xl border transition-colors cursor-pointer ${
                showSearchInput || searchQueryInDoc
                  ? 'bg-[#F97316]/15 text-[#EA580C] border-[#F97316]/40'
                  : 'bg-[#EFE9DF] text-[#6C5E4E] hover:bg-[#E7E0D3] border-[#E0D7C8]'
              }`}
            >
              <Search className="w-4 h-4" />
            </button>

            {/* Zoom Controls */}
            <div className="hidden sm:flex items-center bg-[#EFE9DF] rounded-xl border border-[#E0D7C8] p-0.5 text-xs text-[#3C3227]">
              <button
                id="doc-zoom-out-btn"
                type="button"
                onClick={zoomOut}
                disabled={zoomLevel <= 50}
                className="p-1.5 hover:bg-[#FAF7F2] disabled:opacity-35 rounded-lg transition-colors cursor-pointer"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>

              <button
                type="button"
                onClick={resetZoom}
                className="px-2 py-1 text-[11px] font-mono font-medium hover:text-[#EA580C] cursor-pointer"
                title="Reset Zoom"
              >
                {zoomLevel}%
              </button>

              <button
                id="doc-zoom-in-btn"
                type="button"
                onClick={zoomIn}
                disabled={zoomLevel >= 200}
                className="p-1.5 hover:bg-[#FAF7F2] disabled:opacity-35 rounded-lg transition-colors cursor-pointer"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Top-right: guest CTA or authenticated user menu */}
        <div className="ml-auto flex items-center shrink-0">
          {!isAuthenticated ? (
            <button
              type="button"
              onClick={() => (onOpenAuth ? onOpenAuth() : setShowAuthModal(true))}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white text-xs font-semibold shadow-sm shadow-[#F97316]/30 hover:opacity-95 hover:translate-y-px active:translate-y-0 transition-all cursor-pointer"
            >
              Get Started for Free
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <div className="relative">
              <button
                type="button"
                onClick={() => setIsUserMenuOpen((v) => !v)}
                className="flex items-center gap-2 px-2 py-1 rounded-xl hover:bg-[#FAF7F2] transition-colors cursor-pointer"
              >
                <span className="w-7 h-7 rounded-full bg-gradient-to-br from-[#FF7A00] to-[#E65100] text-white text-[11px] font-bold flex items-center justify-center overflow-hidden">
                  {user?.avatar_url && !user.avatar_url.includes('gravatar.com') ? (
                    <img src={user.avatar_url} alt="" className="w-full h-full object-cover" />
                  ) : (
                    (user?.full_name?.charAt(0) || user?.email?.charAt(0) || '?').toUpperCase()
                  )}
                </span>
                <ChevronDown className="w-3.5 h-3.5 text-[#6C5E4E]" />
              </button>

              {isUserMenuOpen && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setIsUserMenuOpen(false)} />
                  <div className="absolute right-0 z-20 mt-2 w-52 bg-white rounded-2xl border border-[#E5DEC3] shadow-xl p-1.5">
                    <div className="px-3 py-2.5 border-b border-[#EFE9DF] mb-1">
                      <p className="text-sm font-semibold text-[#221C16] truncate">
                        {user?.full_name || 'Account'}
                      </p>
                      <p className="text-[11px] text-[#7A6D5E] truncate">{user?.email}</p>
                      <span className="inline-block mt-1 text-[10px] font-medium px-1.5 py-0.5 rounded-md bg-[#FF7A00]/10 text-[#EA580C]">
                        {user?.is_verified ? 'Verified' : 'Unverified'}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={handleLogout}
                      className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm text-[#3C3227] hover:bg-[#FAF7F2] transition-colors cursor-pointer"
                    >
                      <LogOut className="w-4 h-4 text-[#EA580C]" />
                      Sign out
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </header>

      {/* Inline Search Sub-Bar */}
      <AnimatePresence>
        {showSearchInput && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="px-4 py-2 bg-[#FAF7F2] border-b border-[#E4DCCE] flex items-center gap-2 overflow-hidden shrink-0 z-20"
          >
            <Search className="w-3.5 h-3.5 text-[#8C7D6C]" />
            <input
              type="text"
              value={searchQueryInDoc}
              onChange={(e) => setSearchQueryInDoc(e.target.value)}
              placeholder="Search keyword in this PDF..."
              className="flex-1 bg-transparent text-xs text-[#221C16] outline-none"
              autoFocus
            />
            {searchQueryInDoc && (
              <button
                type="button"
                onClick={() => setSearchQueryInDoc('')}
                className="p-1 text-[#8C7D6C] hover:text-[#221C16] cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Active Citation Jumper Banner (when citation or chunk is highlighted) */}
      <AnimatePresence>
        {activeHighlight && (
          <motion.div
            key={`highlight-banner-${pulseTrigger}`}
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="px-4 py-2 bg-[#FEF3C7] border-b border-[#FDE68A] text-[#92400E] text-xs flex items-center justify-between gap-2 shrink-0 z-20"
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="w-5 h-5 rounded-md bg-[#F97316] text-white flex items-center justify-center text-[10px] font-mono font-bold shrink-0">
                {activeHighlight.citationId ? `[${activeHighlight.citationId}]` : '★'}
              </span>
              <span className="font-semibold truncate">
                {activeHighlight.chunkTitle || 'Retrieved RAG Passage'}
              </span>
              <span className="hidden sm:inline text-[11px] text-[#B45309]">
                • Page {activeHighlight.page} • {Math.round((activeHighlight.confidence || 0.95) * 100)}% Vector Match
              </span>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => {
                  highlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }}
                className="px-2 py-0.5 bg-[#FDE68A] hover:bg-[#FCD34D] text-[#78350F] rounded-md font-semibold text-[11px] transition-colors cursor-pointer"
              >
                Re-Center
              </button>
              <button
                type="button"
                onClick={clearActiveCitation}
                className="p-1 hover:bg-[#FDE68A] rounded-md transition-colors cursor-pointer"
                title="Dismiss Highlight"
              >
                <X className="w-3.5 h-3.5 text-[#92400E]" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Workspace Area (Outline Drawer + PDF Stage) */}
      <div className="flex-1 flex min-h-0 relative">
        {/* Left Outline / Page Thumbnails Sidebar */}
        <AnimatePresence>
          {isOutlineOpen && currentDoc && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 220, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="bg-[#FAF7F2] border-r border-[#E4DCCE] flex flex-col shrink-0 overflow-hidden z-10"
            >
              <div className="p-3 border-b border-[#EDE6DC] flex items-center justify-between text-xs font-bold text-[#6D5E4E]">
                <span>Document Pages</span>
                <span className="font-mono text-[#EA580C]">{currentDoc.totalPages} Pages</span>
              </div>

              <div className="flex-1 overflow-y-auto p-3 space-y-3">
                {currentDoc.pages.map((p) => {
                  const isCurrent = p.pageNumber === activePage;
                  const hasChunk = p.content.some((c) => c.isChunk);

                  return (
                    <button
                      key={`thumb-p-${p.pageNumber}`}
                      type="button"
                      onClick={() => setActivePage(p.pageNumber)}
                      className={`w-full text-left p-2 rounded-xl transition-all border cursor-pointer ${
                        isCurrent
                          ? 'bg-[#F97316]/10 border-[#F97316] ring-1 ring-[#F97316]/30'
                          : 'bg-white border-[#E6DEC3] hover:border-[#D5CBB9]'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[11px] font-bold text-[#2A2219]">Page {p.pageNumber}</span>
                        {hasChunk && (
                          <span className="inline-flex items-center gap-0.5 text-[9px] font-mono px-1 py-0.5 bg-[#FEF3C7] text-[#B45309] rounded-sm font-semibold">
                            Chunk
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-[#7A6C5B] line-clamp-1 font-medium">
                        {p.title || `Section ${p.pageNumber}`}
                      </div>
                      {/* Mini skeleton preview */}
                      <div className="mt-2 h-16 bg-[#F7F3EC] rounded-lg p-1.5 flex flex-col gap-1 border border-[#EDE5D8]">
                        <div className="h-1.5 w-3/4 bg-[#E0D7C9] rounded" />
                        <div className="h-1 w-full bg-[#EAE2D5] rounded" />
                        <div className="h-1 w-5/6 bg-[#EAE2D5] rounded" />
                        {hasChunk && <div className="h-2 w-full bg-[#FDE68A] rounded border border-[#F59E0B]/30" />}
                      </div>
                    </button>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* PDF Stage / Document Viewer Canvas */}
        <div
          ref={documentContainerRef}
          className="flex-1 overflow-y-auto overflow-x-auto p-4 sm:p-8 flex justify-center items-start"
          id="pdf-canvas-scroll-container"
        >
          {/* Empty State: restoring skeleton / guest hero / authed docs skeleton / placeholder / document */}
          {!currentDoc && isSessionRestoring ? (
            /* Neutral skeleton while the session is being restored — never the
               guest hero (prevents the split-second marketing flash). */
            <div className="m-auto flex flex-col items-center justify-center text-center max-w-md p-8">
              <div className="w-16 h-16 rounded-3xl bg-[#E8DFD2] animate-pulse mb-4" />
              <div className="h-3.5 w-44 rounded bg-[#E0D7C9] animate-pulse mb-2" />
              <div className="h-2.5 w-64 rounded bg-[#EAE2D5] animate-pulse" />
            </div>
          ) : !currentDoc && isAuthenticated && isLoadingDocuments ? (
            /* Skeleton for the documents selection stage */
            <div className="m-auto flex flex-col items-center justify-center text-center max-w-md p-8">
              <div className="w-16 h-16 rounded-3xl bg-[#F3EBDF] animate-pulse mb-4" />
              <div className="h-3 w-44 rounded bg-[#E0D7C9] animate-pulse mb-2" />
              <div className="h-2.5 w-56 rounded bg-[#EAE2D5] animate-pulse" />
              <p className="mt-4 text-xs text-[#A59787]">Loading your documents...</p>
            </div>
          ) : !currentDoc && !isAuthenticated ? (
            /* Gradient Welcome Panel (guest) */
            <div className="relative w-full max-w-5xl mx-auto my-2 sm:my-4 rounded-[2rem] overflow-hidden bg-gradient-to-br from-[#FFEDD5] via-[#F6F2EC] to-[#FFD9B3] border border-[#F3DFC4] shadow-[0_30px_60px_-20px_rgba(234,88,12,0.35)]">
              {/* Ambient color orbs */}
              <div aria-hidden className="pointer-events-none absolute inset-0">
                <motion.div
                  className="absolute -top-24 -left-24 w-72 h-72 rounded-full bg-[#FF7A00]/25 blur-3xl"
                  animate={{ x: [0, 24, 0], y: [0, 16, 0], scale: [1, 1.15, 1] }}
                  transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
                />
                <motion.div
                  className="absolute -bottom-28 -right-20 w-80 h-80 rounded-full bg-[#EA580C]/20 blur-3xl"
                  animate={{ x: [0, -28, 0], y: [0, -18, 0], scale: [1, 1.2, 1] }}
                  transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
                />
                <motion.div
                  className="absolute top-1/3 left-1/2 w-56 h-56 rounded-full bg-[#FDBA74]/30 blur-3xl"
                  animate={{ x: [0, 18, 0], y: [0, -22, 0], scale: [1, 1.1, 1] }}
                  transition={{ duration: 16, repeat: Infinity, ease: 'easeInOut' }}
                />
              </div>

              {/* Content */}
              <div className="relative px-6 sm:px-12 py-12 sm:py-16 text-center">
                {/* Badge */}
                <motion.div
                  initial={{ opacity: 0, y: -12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                  className="mb-5 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/80 backdrop-blur text-[#EA580C] text-xs font-semibold border border-[#F5C390] shadow-sm"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Your AI document companion
                  <span className="w-1.5 h-1.5 rounded-full bg-[#F97316] animate-pulse" />
                </motion.div>

                {/* Animated headline */}
                <BlurText
                  text="Ask your"
                  className="font-display text-4xl sm:text-6xl font-bold text-[#221C16] text-center"
                  animateBy="words"
                  direction="top"
                  delay={70}
                  rootMargin="0px"
                  threshold={0}
                />
                <BlurText
                  text="documents anything."
                  className="font-display text-4xl sm:text-6xl font-bold bg-gradient-to-r from-[#FF7A00] via-[#F97316] to-[#EA580C] bg-clip-text text-transparent text-center"
                  animateBy="words"
                  direction="top"
                  delay={90}
                  rootMargin="0px"
                  threshold={0}
                />

                {/* Sub-description */}
                <motion.p
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.35, ease: 'easeOut' }}
                  className="mt-6 mx-auto max-w-xl text-sm sm:text-base text-[#6B5B49] leading-relaxed"
                >
                  Upload any PDF, DOCX, or TXT and have a natural conversation with it.
                  Every answer is <span className="font-semibold text-[#3C3227]">grounded in your sources</span> with
                  clickable citations and in-text highlights you can verify instantly.
                </motion.p>

                {/* Feature glass cards */}
                <div className="mt-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
                  {FEATURES.map((f, idx) => (
                    <motion.div
                      key={f.title}
                      initial={{ opacity: 0, y: 24 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.45 + idx * 0.12, ease: 'easeOut' }}
                      className="group relative rounded-2xl bg-white/75 backdrop-blur-md border border-white/70 p-5 text-left shadow-lg shadow-[#EA580C]/5 hover:shadow-xl hover:shadow-[#F97316]/15 hover:-translate-y-1 transition-all duration-300"
                    >
                      <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#FF7A00] to-[#EA580C] text-white flex items-center justify-center shadow-md shadow-[#F97316]/30 group-hover:scale-110 transition-transform duration-300 mb-4">
                        {f.icon}
                      </div>
                      <h4 className="font-display text-lg font-bold text-[#221C16] mb-1.5">
                        {f.title}
                      </h4>
                      <p className="text-xs text-[#7A6D5E] leading-relaxed">{f.desc}</p>
                    </motion.div>
                  ))}
                </div>

                {/* CTA row */}
                <motion.div
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.9, ease: 'easeOut' }}
                  className="mt-12 flex flex-col sm:flex-row items-center justify-center gap-3"
                >
                  <button
                    type="button"
                    onClick={() => (onOpenAuth ? onOpenAuth() : setShowAuthModal(true))}
                    className="group inline-flex items-center gap-2.5 px-8 py-3.5 rounded-2xl bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white text-sm font-semibold shadow-lg shadow-[#F97316]/40 hover:shadow-xl hover:shadow-[#F97316]/50 hover:-translate-y-0.5 active:translate-y-0 transition-all cursor-pointer"
                  >
                    Get Started for Free
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                  <span className="text-xs text-[#8C7D6C]">No credit card required · 2 min setup</span>
                </motion.div>
              </div>
            </div>
          ) : !currentDoc ? (
            /* Placeholder for authenticated users with no documents */
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="m-auto text-center max-w-md p-8 bg-white/90 backdrop-blur-md rounded-3xl border border-[#E5DEC3] shadow-lg shadow-[#3F3323]/5"
            >
              <div className="w-16 h-16 rounded-3xl bg-[#F97316]/10 text-[#EA580C] flex items-center justify-center mx-auto mb-4 shadow-inner">
                <BookOpen className="w-8 h-8" />
              </div>
              <h3 className="font-display text-2xl font-bold text-[#221C16] mb-2">
                No documents yet
              </h3>
              <p className="text-xs sm:text-sm text-[#7A6D5E] leading-relaxed">
                Upload a document from the chat to start asking grounded questions with verifiable citations.
              </p>
            </motion.div>
          ) : (
            /* High Fidelity Simulated PDF Document Sheet */
            <motion.div
              key={`${currentDoc.id}-p-${activePage}`}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              style={{
                width: `${Math.round(zoomLevel * 7.5)}px`,
                maxWidth: zoomLevel <= 100 ? '100%' : 'none'
              }}
              className="bg-white text-[#1F1914] rounded-2xl shadow-xl shadow-[#3F3323]/10 border border-[#DDD4C3] p-8 sm:p-12 relative min-h-[900px] flex flex-col justify-between transition-all"
            >
              {/* Top Document Header Bar */}
              <div>
                <div className="flex items-center justify-between pb-4 border-b border-[#EAE2D5] text-[11px] font-mono text-[#8C7E6E] tracking-wider mb-6">
                  <span>{currentPageData?.sectionHeader || 'TECHNICAL REPORT'}</span>
                  <span className="font-bold text-[#EA580C]">PAGE {activePage} OF {currentDoc.totalPages}</span>
                </div>

                {/* Page Content Body */}
                <div className="space-y-6">
                  {currentPageData?.content.map((block, idx) => {
                    const isHighlightedChunk =
                      activeHighlight &&
                      activeHighlight.page === activePage &&
                      (block.citationId === activeHighlight.citationId || (block.isChunk && activeCitationId === block.citationId));

                    if (block.type === 'heading') {
                      return (
                        <h2
                          key={`heading-${idx}`}
                          className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-[#16110C] leading-snug border-b border-[#F0EAE0] pb-3"
                        >
                          {block.text}
                        </h2>
                      );
                    }

                    if (block.type === 'subheading') {
                      return (
                        <h3
                          key={`subheading-${idx}`}
                          className="text-base sm:text-lg font-bold text-[#2A2118] mt-4 font-display"
                        >
                          {block.text}
                        </h3>
                      );
                    }

                    if (block.type === 'callout') {
                      return (
                        <div
                          key={`callout-${idx}`}
                          className="p-4 rounded-xl bg-[#FAF6F0] border-l-4 border-[#F97316] text-xs sm:text-sm text-[#3E3326] leading-relaxed italic font-editorial"
                        >
                          {block.text}
                        </div>
                      );
                    }

                    if (block.type === 'equation') {
                      return (
                        <div
                          key={`eq-${idx}`}
                          className="my-3 p-4 rounded-xl bg-[#F8F4ED] border border-[#E5DEC3] text-center font-mono text-xs sm:text-sm text-[#5D2B0C] tracking-wide shadow-inner"
                        >
                          {block.text}
                        </div>
                      );
                    }

                    if (block.type === 'table') {
                      return (
                        <div
                          key={`table-${idx}`}
                          className="my-4 overflow-x-auto rounded-xl border border-[#E5DEC3] bg-[#FAF8F5]"
                        >
                          {block.caption && (
                            <div className="px-4 py-2 bg-[#F3EDE3] text-[11px] font-semibold text-[#5A4C3E] border-b border-[#E5DEC3]">
                              {block.caption}
                            </div>
                          )}
                          <div className="p-3 text-xs font-mono space-y-1.5">
                            {block.items?.map((row, rIdx) => (
                              <div
                                key={`row-${rIdx}`}
                                className={`p-1.5 rounded flex items-center justify-between ${
                                  rIdx === 0
                                    ? 'bg-[#EAE2D5] font-bold text-[#2A2118]'
                                    : rIdx === 2
                                    ? 'bg-[#FEF3C7] text-[#92400E] font-semibold'
                                    : 'text-[#4A3E31]'
                                }`}
                              >
                                <span>{row}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    }

                    if (block.type === 'bullet-list') {
                      return (
                        <ul key={`list-${idx}`} className="space-y-2 text-xs sm:text-sm text-[#352B21] pl-5 list-disc">
                          {block.items?.map((item, lIdx) => (
                            <li key={`item-${lIdx}`} className="leading-relaxed">
                              {item}
                            </li>
                          ))}
                        </ul>
                      );
                    }

                    // Standard Paragraph OR Retrieved Chunk Box
                    return (
                      <div
                        key={`p-${idx}`}
                        ref={isHighlightedChunk ? highlightRef : null}
                        className={`relative transition-all duration-300 rounded-2xl ${
                          isHighlightedChunk
                            ? 'p-4 bg-gradient-to-r from-[#FEF3C7] to-[#FED7AA]/60 border-2 border-[#F97316] ring-4 ring-[#F97316]/20 shadow-lg highlight-pulse'
                            : block.isChunk
                            ? 'p-3 bg-[#FAF7F0] border border-[#E9E2D5] hover:border-[#F97316]/40 cursor-pointer rounded-xl'
                            : ''
                        }`}
                      >
                        {/* Chunk Badge Header */}
                        {(isHighlightedChunk || block.isChunk) && (
                          <div className="flex items-center justify-between gap-2 mb-2">
                            <div className="flex items-center gap-1.5">
                              <span className={`px-2 py-0.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider flex items-center gap-1 ${
                                isHighlightedChunk
                                  ? 'bg-[#F97316] text-white shadow-xs'
                                  : 'bg-[#EFE9DF] text-[#7C3AED]'
                              }`}>
                                <Sparkles className="w-2.5 h-2.5" />
                                {block.citationId ? `Citation [${block.citationId}] Chunk` : 'Retrieved Vector Chunk'}
                              </span>
                              {block.chunkMeta && (
                                <span className="text-[11px] font-semibold text-[#4A3E31]">
                                  {block.chunkMeta.label}
                                </span>
                              )}
                            </div>

                            <span className="text-[10px] font-mono font-bold text-[#EA580C] bg-white/90 px-1.5 py-0.5 rounded border border-[#E8E0D2]">
                              Score: {Math.round((block.chunkMeta?.confidence || 0.94) * 100)}% MIPS
                            </span>
                          </div>
                        )}

                        <p className={`text-xs sm:text-sm leading-relaxed ${
                          isHighlightedChunk
                            ? 'font-medium text-[#291F16]'
                            : 'text-[#362C22]'
                        }`}>
                          {block.text}
                        </p>

                        {/* Interactive Click to Jump from document */}
                        {block.isChunk && !isHighlightedChunk && (
                          <div className="mt-2 text-right">
                            <button
                              type="button"
                              onClick={() => {
                                if (block.citationId) {
                                  // Jump to this chunk
                                  useViewerStore.getState().jumpToCitation({
                                    id: block.citationId,
                                    docId: currentDoc.id,
                                    docTitle: currentDoc.title,
                                    page: activePage,
                                    snippet: block.text || '',
                                    chunkTitle: block.chunkMeta?.label || 'Vector Chunk',
                                    confidence: block.chunkMeta?.confidence || 0.94,
                                    score: 0.91,
                                    boundingBox: { top: 30, left: 6, width: 88, height: 18 }
                                  });
                                }
                              }}
                              className="text-[10px] font-semibold text-[#EA580C] hover:underline cursor-pointer"
                            >
                              Highlight in Active Session →
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Document Page Footer */}
              <div className="mt-12 pt-4 border-t border-[#EAE2D5] flex items-center justify-between text-[11px] text-[#9A8D7E] font-mono">
                <span className="truncate max-w-[240px]">{currentDoc.title}</span>
                <span>Page {activePage}</span>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};
