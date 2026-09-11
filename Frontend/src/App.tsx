import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChatPane } from './components/ChatPane';
import { DocumentViewer } from './components/DocumentViewer';
import { UploadModal } from './components/UploadModal';
import { AuthModal, AuthMode, ResolveMergeInfo } from './components/AuthModal';
import { VerifyEmailPage } from './pages/VerifyEmailPage';
import { ForgotPasswordPage } from './pages/ForgotPasswordPage';
import { ResetPasswordPage } from './pages/ResetPasswordPage';
import { useViewerStore } from './store/useViewerStore';
import { ConversationSidebar } from './components/ConversationSidebar';
import { me } from './api/auth';
import { listDocuments, toRagDocument } from './api/documents';
import { listConversations } from './api/chat';
import { loadDocSelection } from './utils/docSelection';
import {
  currentConversationSlug,
  goHome,
  isHomeRoute
} from './utils/navigation';
import { selectConversation } from './utils/chatNavigation';
import { X, BookOpen } from 'lucide-react';

interface AuthContextState {
  mode: AuthMode;
  resetToken: string;
  merge: ResolveMergeInfo | null;
}

function readPostOAuthContext(): AuthContextState | null {
  const pathname = window.location.pathname;
  const searchParams = new URLSearchParams(window.location.search);
  const clean = () => {
    window.history.replaceState({}, document.title, pathname.split('/auth')[0] || '/');
  };

  if (pathname === '/auth/google/callback') {
    // Session cookies were already set by the backend; restore below.
    clean();
    return { mode: 'signin', resetToken: '', merge: null };
  }
  if (pathname === '/auth/merge') {
    const ctx: AuthContextState = {
      mode: 'merge',
      resetToken: '',
      merge: {
        token: searchParams.get('token') || '',
        email: searchParams.get('email') || '',
        name: searchParams.get('name') || undefined,
        avatar: searchParams.get('avatar') || undefined,
      },
    };
    clean();
    return ctx;
  }
  return null;
}

export default function App() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const { isViewerOpen, setViewerOpen, showAuthModal, setShowAuthModal, isAuthenticated } =
    useViewerStore();

  // After a Google redirect a fresh tab may not have the store; drive the
  // modal state locally so it can be opened from the URL.
  const [authCtx, setAuthCtx] = useState<AuthContextState>({
    mode: 'signin',
    resetToken: '',
    merge: null,
  });
  const [authCtxKey, setAuthCtxKey] = useState(0);

  const restoreSession = async () => {
    let user;
    try {
      user = await me();
    } catch {
      // No valid session — signed out until the user signs in.
      useViewerStore.getState().setUser(null);
      setShowAuthModal(false);
      return;
    } finally {
      // Guests may now see the marketing welcome; authed users see their
      // workspace (documents/conversations load via the isAuthenticated
      // effect below). No more splash of guest animations on refresh.
      useViewerStore.getState().setSessionRestoring(false);
    }
    useViewerStore.getState().setUser(user);
    setShowAuthModal(false);
  };

  useEffect(() => {
    const ctx = readPostOAuthContext();
    if (ctx) {
      setAuthCtx(ctx);
      setAuthCtxKey((k) => k + 1);
      if (ctx.mode === 'merge' || ctx.mode === 'reset') {
        setShowAuthModal(true);
      } else {
        // plain Google success callback: restore the session from cookies
        restoreSession();
      }
    } else {
      // Cold start: restore an existing session (if any).
      restoreSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Single source of truth for authed data bootstrap (covers both the session
  // restored on mount/refresh and a fresh sign-in via the modal).
  useEffect(() => {
    const store = useViewerStore.getState();
    const u = store.user;

    if (!u) {
      store.setConversations([]);
      store.setActiveChatId(null);
      store.setIsLoadingConversations(false);
      store.setIsLoadingDocuments(false);
      store.setIsLoadingMessages(false);
      if (currentConversationSlug()) goHome();
      return;
    }

    store.setSelectedDocIds(loadDocSelection(u.user_id));
    store.setIsLoadingDocuments(true);
    store.setIsLoadingConversations(true);

    const userId = u.user_id;
    Promise.allSettled([listDocuments(), listConversations()]).then(([docsRes, convosRes]) => {
      // Drop stale results if the user signed out / switched mid-flight.
      if (useViewerStore.getState().user?.user_id !== userId) return;

      store.setIsLoadingDocuments(false);
      store.setIsLoadingConversations(false);

      if (docsRes.status === 'fulfilled') {
        store.setDocuments(docsRes.value.map((d) => toRagDocument(d)));
      }

      const convos = convosRes.status === 'fulfilled' ? convosRes.value : [];
      store.setConversations(convos);

      // Deep-link: /c/<chat_id> selects that conversation once loaded.
      const slug = currentConversationSlug();
      if (slug) {
        if (convos.some((c) => c.chat_id === slug)) {
          selectConversation(slug);
        } else {
          goHome();
        }
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated]);

  // Keep the workspace in sync with browser back/forward (URL is the source
  // of truth for the active conversation).
  useEffect(() => {
    const onPopState = () => {
      const store = useViewerStore.getState();
      const slug = currentConversationSlug();
      if (slug) {
        if (store.isAuthenticated) {
          selectConversation(slug);
        } else {
          goHome();
        }
      } else if (isHomeRoute()) {
        store.clearChat();
      }
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, [isAuthenticated]);

  const openAuth = (mode: AuthMode = 'signin') => {
    setAuthCtx({ mode, resetToken: '', merge: null });
    setAuthCtxKey((k) => k + 1);
    setShowAuthModal(true);
  };

  // Standalone auth pages (no router; dispatched off window.location.pathname).
  const route = window.location.pathname;
  if (route === '/verify') return <VerifyEmailPage />;
  if (route === '/forgot-password') return <ForgotPasswordPage />;
  if (route === '/reset-password') return <ResetPasswordPage />;

  return (
    <div className="flex h-screen w-screen bg-[#F6F2EC] text-[#221C16] overflow-hidden select-none font-sans" id="rag-app-root">
      {/* Desktop & Tablet Side-by-Side Split Workspace */}
      <div className="flex flex-1 h-full w-full overflow-hidden">
        {/* Left Pane (approx 40% width on desktop, 100% on small screens) */}
        <section
          aria-label="Conversational AI Chat"
          className="w-full lg:w-[42%] xl:w-[38%] h-full flex flex-row shrink-0 z-10 overflow-hidden"
        >
          <ConversationSidebar />
          <ChatPane onOpenUploadModal={() => setIsUploadModalOpen(true)} onOpenAuth={openAuth} />
        </section>

        {/* Right Pane (approx 60% width on desktop, hidden by default on mobile until opened) */}
        <section
          aria-label="Document PDF Viewer"
          className="hidden lg:flex flex-1 h-full flex-col min-w-0"
        >
          <DocumentViewer onOpenUploadModal={() => setIsUploadModalOpen(true)} onOpenAuth={openAuth} />
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
              <DocumentViewer onOpenUploadModal={() => setIsUploadModalOpen(true)} onOpenAuth={openAuth} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Document Upload Modal */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
      />

      {/* Login / Signup / Reset / Merge Modal */}
      <AuthModal
        key={authCtxKey}
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        initialMode={authCtx.mode}
        resetToken={authCtx.resetToken}
        merge={authCtx.merge}
      />
    </div>
  );
}
