import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Plus, Trash2, MessageSquare, History } from 'lucide-react';
import { useViewerStore } from '../store/useViewerStore';
import { deleteConversation } from '../api/chat';
import type { BackendConversation } from '../api/chat';
import { selectConversation } from '../utils/chatNavigation';
import { goToConversation, goHome, isHomeRoute } from '../utils/navigation';

function conversationTitle(c: BackendConversation): string {
  return c.title?.trim() || 'Untitled conversation';
}

function timeLabel(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export const ConversationSidebar: React.FC = () => {
  const {
    isAuthenticated,
    activeChatId,
    conversations,
    isLoadingConversations,
    setConversations,
    clearChat,
  } = useViewerStore();

  const handleSelect = (chatId: string) => {
    if (chatId === activeChatId) return;
    goToConversation(chatId);
    selectConversation(chatId);
  };

  const handleNewChat = () => {
    // Home is always a brand-new chat. If we are inside a conversation, open
    // the home page; if we are already on the home page, do nothing.
    if (!isHomeRoute()) goHome();
    clearChat();
  };

  const handleDelete = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteConversation(chatId);
      if (activeChatId === chatId) {
        if (!isHomeRoute()) goHome();
        clearChat();
      }
      setConversations(conversations.filter((c) => c.chat_id !== chatId));
    } catch {
      // ignore
    }
  };

  if (!isAuthenticated) return null;

  return (
    <div className="w-52 shrink-0 h-full bg-[#EFE8DD] border-r border-[#DDD2C0] flex flex-col">
      <div className="px-3 pt-3 pb-2">
        <button
          type="button"
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-[#221C16] text-[#F9F6F0] hover:bg-[#3A3229] transition-colors cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" />
          New Chat
        </button>
      </div>

      <div className="px-3 pb-1.5 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-[#8A7B6B]">
        <History className="w-3 h-3" />
        Conversations
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
        {isLoadingConversations && conversations.length === 0 ? (
          <div className="px-1 space-y-2" aria-label="Loading conversations">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex items-start gap-2 px-2.5 py-2">
                <div className="w-3.5 h-3.5 mt-0.5 rounded-full bg-[#D8CCB8]/60 animate-pulse shrink-0" />
                <div className="flex-1 space-y-1.5">
                  <div
                    className="h-2.5 rounded bg-[#D8CCB8]/60 animate-pulse"
                    style={{ width: `${Math.min(92, 58 + ((i * 13) % 34))}%` }}
                  />
                  <div className="h-2 w-14 rounded bg-[#D8CCB8]/40 animate-pulse" />
                </div>
              </div>
            ))}
          </div>
        ) : conversations.length === 0 ? (
          <p className="px-2 py-4 text-[11px] text-[#A59787] text-center leading-relaxed">
            No conversations yet.
            <br />
            Start a new chat to ask your documents.
          </p>
        ) : (
          conversations.map((c) => {
            const active = c.chat_id === activeChatId;
            return (
              <motion.button
                key={c.chat_id}
                type="button"
                onClick={() => handleSelect(c.chat_id)}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                className={`w-full group flex items-start gap-2 px-2.5 py-2 rounded-xl text-left transition-colors cursor-pointer ${
                  active ? 'bg-[#F97316]/15 text-[#221C16]' : 'text-[#3E342A] hover:bg-[#E6DECF]'
                }`}
              >
                <MessageSquare
                  className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${
                    active ? 'text-[#EA580C]' : 'text-[#9A8D7E] group-hover:text-[#EA580C]'
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-semibold truncate leading-snug">
                    {conversationTitle(c)}
                  </p>
                  <p className="text-[10px] text-[#8A7B6B]">{timeLabel(c.updated_at)}</p>
                </div>
                <AnimatePresence>
                  {active && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.5 }}
                      onClick={(e) => handleDelete(c.chat_id, e as React.MouseEvent)}
                      className="p-1 rounded-md text-[#A59787] hover:text-[#DC2626] hover:bg-white/70 cursor-pointer"
                      title="Delete conversation"
                    >
                      <Trash2 className="w-3 h-3" />
                    </motion.span>
                  )}
                </AnimatePresence>
              </motion.button>
            );
          })
        )}
      </div>
    </div>
  );
};
