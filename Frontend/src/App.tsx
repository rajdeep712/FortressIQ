import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChatPane } from './components/ChatPane';
import { DocumentViewer } from './components/DocumentViewer';
import { UploadModal } from './components/UploadModal';
import { useViewerStore } from './store/useViewerStore';
import { X, BookOpen, MessageSquare } from 'lucide-react';

export default function App() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const { isViewerOpen, setViewerOpen } = useViewerStore();

  return (
    <div className="flex h-screen w-screen bg-[#F6F2EC] text-[#221C16] overflow-hidden select-none font-sans" id="rag-app-root">
      {/* Desktop & Tablet Side-by-Side Split Workspace */}
      <div className="flex flex-1 h-full w-full overflow-hidden">
        {/* Left Pane (approx 40% width on desktop, 100% on small screens) */}
        <section
          aria-label="Conversational AI Chat"
          className="w-full lg:w-[42%] xl:w-[38%] h-full flex flex-col shrink-0 z-10"
        >
          <ChatPane onOpenUploadModal={() => setIsUploadModalOpen(true)} />
        </section>

        {/* Right Pane (approx 60% width on desktop, hidden by default on mobile until opened) */}
        <section
          aria-label="Document PDF Viewer"
          className="hidden lg:flex flex-1 h-full flex-col min-w-0"
        >
          <DocumentViewer onOpenUploadModal={() => setIsUploadModalOpen(true)} />
        </section>
      </div>

      {/* Mobile Slide-Over Viewer Drawer */}
      <AnimatePresence>
        {isViewerOpen && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 300 }}
            className="fixed inset-0 z-50 lg:hidden flex flex-col bg-[#EDE6DC]"
          >
            {/* Mobile Viewer Header Bar */}
            <div className="px-4 py-2.5 bg-[#FAF7F2] border-b border-[#E4DCCE] flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-[#F97316]/15 text-[#EA580C] flex items-center justify-center">
                  <BookOpen className="w-3.5 h-3.5" />
                </div>
                <span className="text-xs font-bold text-[#221C16]">Grounded Document Viewer</span>
              </div>

              <button
                type="button"
                onClick={() => setViewerOpen(false)}
                className="p-1.5 rounded-lg bg-[#EFE9DF] text-[#6C5E4E] hover:bg-[#E7E0D3] transition-colors flex items-center gap-1 text-xs font-semibold cursor-pointer"
              >
                <X className="w-4 h-4" />
                <span>Back to Chat</span>
              </button>
            </div>

            <div className="flex-1 min-h-0">
              <DocumentViewer onOpenUploadModal={() => setIsUploadModalOpen(true)} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Document Upload Modal */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
      />
    </div>
  );
}
