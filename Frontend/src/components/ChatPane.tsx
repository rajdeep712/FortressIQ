import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Send,
  Sparkles,
  RefreshCw,
  Layers,
  FileText,
  Copy,
  Check,
  ChevronDown,
  Cpu,
  Zap,
  ShieldCheck,
  PlusCircle,
  Clock,
  BookOpen
} from 'lucide-react';
import { useViewerStore } from '../store/useViewerStore';
import { PRESET_PROMPTS } from '../data/mockDocuments';
import { generateRAGResponse } from '../data/ragEngine';
import { MarkdownRenderer } from './MarkdownRenderer';
import { Message } from '../types';

interface ChatPaneProps {
  onOpenUploadModal?: () => void;
}

export const ChatPane: React.FC<ChatPaneProps> = ({ onOpenUploadModal }) => {
  const [inputValue, setInputValue] = useState('');
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [showPresetsMenu, setShowPresetsMenu] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    documents,
    isGenerating,
    selectedModel,
    addMessage,
    setIsGenerating,
    clearChat,
    setSelectedModel,
    setViewerOpen,
    isViewerOpen,
    activeCitationId,
    jumpToCitation
  } = useViewerStore();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  // Handle submitting user message & running mock RAG generation
  const handleSendMessage = (textToSend?: string) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isGenerating) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    addMessage(userMessage);
    setInputValue('');
    setIsGenerating(true);

    // Simulate DPR vector retrieval & generation pipeline with progressive steps
    setTimeout(() => {
      const ragResult = generateRAGResponse(text, documents);

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: ragResult.content,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        citations: ragResult.citations,
        model: selectedModel,
        retrievalLatencyMs: ragResult.retrievalLatencyMs,
        tokensCount: ragResult.tokensCount
      };

      addMessage(assistantMessage);
      setIsGenerating(false);

      // If citations exist, jump to the first one automatically
      if (ragResult.citations.length > 0) {
        jumpToCitation(ragResult.citations[0]);
      }
    }, 750);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMessageId(id);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const availableModels = [
    { name: 'Mistral Large 2', desc: '128k context • SOTA reasoning & code' },
    { name: 'Mistral NeMo 12B', desc: 'Fast multimodal & efficient RAG' },
    { name: 'Codestral 22B', desc: 'Specialized for code repositories' }
  ];

  return (
    <div className="flex flex-col h-full bg-[#F6F2EC] border-r border-[#E5DEC3] relative select-text" id="chat-pane-container">
      {/* Header */}
      <header className="px-5 py-4 bg-[#F9F6F0]/90 backdrop-blur-md border-b border-[#E8E1D5] flex items-center justify-between shrink-0 z-20">
        <div className="flex items-center gap-3">
          {/* Mistral-inspired signature orange flame/pixel badge */}
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#FF7A00] to-[#E65100] flex items-center justify-center shadow-md shadow-[#F97316]/25 ring-1 ring-[#FF9E40]/30 text-white font-bold text-lg font-mono">
            <span className="leading-none tracking-tighter">M</span>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-display text-xl font-bold tracking-tight text-[#221C16]">
                Mistral RAG Studio
              </h1>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-[#F97316]/15 text-[#C2410C] border border-[#F97316]/30">
                v2.5 DPR
              </span>
            </div>
            <p className="text-xs text-[#7A6E60] flex items-center gap-1.5 mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>{documents.length} papers indexed in FAISS memory</span>
            </p>
          </div>
        </div>

        {/* Header Action Buttons */}
        <div className="flex items-center gap-2">
          {/* Model Selector Dropdown */}
          <div className="relative">
            <button
              id="model-selector-btn"
              type="button"
              onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#EFE9DE] hover:bg-[#E7E0D3] text-[#3E342A] rounded-xl border border-[#E0D7C8] transition-colors cursor-pointer"
            >
              <Cpu className="w-3.5 h-3.5 text-[#EA580C]" />
              <span className="font-semibold hidden sm:inline">{selectedModel}</span>
              <ChevronDown className="w-3 h-3 text-[#7A6E60]" />
            </button>

            <AnimatePresence>
              {modelDropdownOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 6, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 4, scale: 0.95 }}
                  transition={{ duration: 0.12 }}
                  className="absolute right-0 mt-2 w-64 p-1.5 bg-[#FAF7F2] rounded-2xl border border-[#E5DEC3] shadow-xl z-50 text-left"
                >
                  <div className="px-3 py-1.5 text-[11px] font-bold text-[#8C7E6E] uppercase tracking-wider">
                    Select Inference Model
                  </div>
                  {availableModels.map((m) => (
                    <button
                      key={m.name}
                      onClick={() => {
                        setSelectedModel(m.name);
                        setModelDropdownOpen(false);
                      }}
                      className={`w-full text-left p-2.5 rounded-xl text-xs transition-colors flex items-start justify-between cursor-pointer ${
                        selectedModel === m.name
                          ? 'bg-[#F97316] text-white'
                          : 'hover:bg-[#EFE9DE] text-[#2C241D]'
                      }`}
                    >
                      <div>
                        <div className="font-semibold">{m.name}</div>
                        <div className={`text-[11px] ${selectedModel === m.name ? 'text-white/80' : 'text-[#7A6E60]'}`}>
                          {m.desc}
                        </div>
                      </div>
                      {selectedModel === m.name && <Check className="w-4 h-4 shrink-0 mt-0.5" />}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Clear / Reset Chat */}
          <button
            id="clear-chat-btn"
            type="button"
            onClick={clearChat}
            title="Reset conversation"
            className="p-2 text-[#7A6E60] hover:text-[#221C16] hover:bg-[#EFE9DE] rounded-xl border border-transparent hover:border-[#E0D7C8] transition-colors cursor-pointer"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          {/* Mobile Toggle Document Viewer */}
          <button
            id="mobile-viewer-toggle-btn"
            type="button"
            onClick={() => setViewerOpen(!isViewerOpen)}
            className="lg:hidden p-2 bg-[#F97316]/15 text-[#EA580C] border border-[#F97316]/30 rounded-xl text-xs font-semibold flex items-center gap-1 cursor-pointer"
          >
            <BookOpen className="w-4 h-4" />
            <span>PDF</span>
          </button>
        </div>
      </header>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-5" id="chat-messages-scroll-area">
        {messages.map((message) => {
          const isUser = message.role === 'user';

          return (
            <motion.div
              key={message.id}
              id={`chat-msg-${message.id}`}
              initial={{ opacity: 0, y: 14, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{
                type: 'spring',
                stiffness: 380,
                damping: 26
              }}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
            >
              {/* Message Label/Timestamp */}
              <div className="flex items-center gap-2 mb-1.5 px-1 text-[11px] text-[#8A7C6D]">
                {!isUser && (
                  <div className="flex items-center gap-1 font-medium text-[#D95D0F]">
                    <Sparkles className="w-3 h-3 text-[#EA580C]" />
                    <span className="font-mono text-[10px] font-semibold">{message.model || 'Mistral RAG'}</span>
                  </div>
                )}
                <span>{message.timestamp}</span>
              </div>

              {/* Message Bubble Container */}
              <div
                className={`max-w-[92%] sm:max-w-[85%] rounded-3xl p-4 sm:p-5 transition-all ${
                  isUser
                    ? 'bg-[#231E19] text-[#F9F6F0] rounded-tr-sm shadow-md'
                    : 'bg-white border border-[#E7DFC0]/80 text-[#2D2620] rounded-tl-sm shadow-sm'
                }`}
              >
                {isUser ? (
                  <p className="text-sm font-medium whitespace-pre-wrap leading-relaxed text-[#F9F6F0]">
                    {message.content}
                  </p>
                ) : (
                  <div>
                    <MarkdownRenderer content={message.content} citations={message.citations} />

                    {/* Metadata & Cited Sources Bar */}
                    {message.citations && message.citations.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-[#F0EBE1] flex flex-wrap items-center justify-between gap-2 text-xs">
                        <div className="flex items-center flex-wrap gap-1.5">
                          <span className="text-[11px] font-semibold text-[#7E6F5E] flex items-center gap-1">
                            <FileText className="w-3.5 h-3.5 text-[#EA580C]" />
                            Sources Cited:
                          </span>
                          {message.citations.map((cite) => (
                            <button
                              key={`cite-pill-${cite.id}`}
                              type="button"
                              onClick={() => jumpToCitation(cite)}
                              className={`inline-flex items-center gap-1 px-2 py-0.5 text-[11px] rounded-lg border font-mono transition-all cursor-pointer ${
                                activeCitationId === cite.id
                                  ? 'bg-[#F97316] text-white border-[#EA580C] shadow-xs'
                                  : 'bg-[#FAF7F2] text-[#4A3D30] hover:bg-[#F3ECE0] border-[#E5DEC3]'
                              }`}
                            >
                              <span className="font-bold">[{cite.id}]</span>
                              <span className="truncate max-w-[120px]">{cite.docTitle.split(':')[0]}</span>
                              <span className="text-[10px] opacity-75">p.{cite.page}</span>
                            </button>
                          ))}
                        </div>

                        {/* Latency & Copy Button */}
                        <div className="flex items-center gap-2 text-[11px] text-[#918272] ml-auto">
                          {message.retrievalLatencyMs && (
                            <span className="inline-flex items-center gap-1 font-mono text-[10px]">
                              <Clock className="w-3 h-3 text-[#A89A8B]" />
                              {message.retrievalLatencyMs}ms
                            </span>
                          )}
                          <button
                            type="button"
                            onClick={() => handleCopy(message.id, message.content)}
                            title="Copy response"
                            className="p-1 hover:bg-[#F5EFE4] text-[#7A6E60] hover:text-[#221C16] rounded-md transition-colors cursor-pointer"
                          >
                            {copiedMessageId === message.id ? (
                              <Check className="w-3.5 h-3.5 text-emerald-600" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}

        {/* Loading / Generating State Animation */}
        {isGenerating && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-start"
          >
            <div className="flex items-center gap-2 mb-1.5 px-1 text-[11px] text-[#D95D0F]">
              <Sparkles className="w-3 h-3 animate-spin text-[#EA580C]" />
              <span className="font-semibold font-mono">{selectedModel}</span>
            </div>

            <div className="bg-white border border-[#E7DFC0] rounded-3xl rounded-tl-sm p-4 shadow-sm flex flex-col gap-2 min-w-[280px]">
              <div className="flex items-center gap-2 text-xs font-semibold text-[#4F4235]">
                <div className="w-2 h-2 rounded-full bg-[#F97316] animate-ping" />
                <span>Searching DPR index & extracting passages...</span>
              </div>
              <div className="h-1.5 w-full bg-[#F3ECE0] rounded-full overflow-hidden">
                <motion.div
                  initial={{ x: '-100%' }}
                  animate={{ x: '100%' }}
                  transition={{ repeat: Infinity, duration: 1.2, ease: 'easeInOut' }}
                  className="h-full w-1/2 bg-gradient-to-r from-[#FF7A00] to-[#E65100] rounded-full"
                />
              </div>
              <span className="text-[11px] text-[#8C7D6C]">MIPS nearest-neighbor cosine matching (k=5)</span>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Prompt Cards / Quick Starters */}
      {messages.length <= 1 && (
        <div className="px-5 pb-2 shrink-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-[#8A7B6B] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#F97316]" />
              Suggested Research Questions
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {PRESET_PROMPTS.map((prompt) => (
              <motion.button
                key={prompt.id}
                type="button"
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleSendMessage(prompt.query)}
                className="text-left p-3 rounded-2xl bg-white hover:bg-[#FFFDF9] border border-[#E8E0D1] hover:border-[#F97316]/50 shadow-xs hover:shadow-md transition-all group cursor-pointer"
              >
                <div className="flex items-start gap-2.5">
                  <div className="p-1.5 rounded-xl bg-[#F97316]/10 text-[#EA580C] group-hover:bg-[#F97316] group-hover:text-white transition-colors shrink-0 mt-0.5">
                    {prompt.icon === 'Cpu' && <Cpu className="w-3.5 h-3.5" />}
                    {prompt.icon === 'Layers' && <Layers className="w-3.5 h-3.5" />}
                    {prompt.icon === 'Zap' && <Zap className="w-3.5 h-3.5" />}
                    {prompt.icon === 'ShieldCheck' && <ShieldCheck className="w-3.5 h-3.5" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-xs font-bold text-[#2A2219] group-hover:text-[#EA580C] transition-colors line-clamp-1">
                      {prompt.title}
                    </h4>
                    <p className="text-[11px] text-[#786A5A] line-clamp-2 mt-0.5 leading-snug">
                      {prompt.query}
                    </p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        </div>
      )}

      {/* Floating Pill-shaped Input Area */}
      <div className="p-4 bg-gradient-to-t from-[#F6F2EC] via-[#F6F2EC]/90 to-transparent shrink-0">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="relative bg-white rounded-3xl border border-[#E5DEC3] shadow-lg shadow-[#3F3323]/5 p-2 flex items-center gap-2 focus-within:ring-2 focus-within:ring-[#F97316]/40 focus-within:border-[#F97316] transition-all"
        >
          {/* Quick Upload / Add Doc Button */}
          <button
            id="chat-upload-doc-btn"
            type="button"
            onClick={onOpenUploadModal}
            title="Index new document into vector store"
            className="p-2.5 text-[#7E6F5E] hover:text-[#EA580C] hover:bg-[#F8F3EA] rounded-full transition-colors cursor-pointer shrink-0"
          >
            <PlusCircle className="w-5 h-5" />
          </button>

          {/* Text Area */}
          <textarea
            id="chat-input-textarea"
            rows={1}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about indexed documents..."
            className="flex-1 bg-transparent text-sm text-[#231E19] placeholder-[#9A8D7E] resize-none outline-none py-2 px-1 max-h-28 overflow-y-auto leading-relaxed"
          />

          {/* Send Button */}
          <motion.button
            id="chat-send-btn"
            type="submit"
            disabled={!inputValue.trim() || isGenerating}
            whileHover={{ scale: 1.06 }}
            whileTap={{ scale: 0.94 }}
            className={`p-3 rounded-full flex items-center justify-center transition-all cursor-pointer shrink-0 ${
              inputValue.trim() && !isGenerating
                ? 'bg-gradient-to-r from-[#FF7A00] to-[#EA580C] text-white shadow-md shadow-[#F97316]/30'
                : 'bg-[#EFE9DF] text-[#A89C8E] cursor-not-allowed'
            }`}
          >
            <Send className="w-4 h-4" />
          </motion.button>
        </form>

        {/* Input Footer Subtext */}
        <div className="mt-2 flex items-center justify-between text-[11px] text-[#8E8070] px-3">
          <span className="flex items-center gap-1">
            <span>Press</span>
            <kbd className="px-1.5 py-0.5 rounded bg-[#EAE3D6] text-[#4A3E31] text-[10px] font-mono border border-[#DDD3C2]">
              Return
            </kbd>
            <span>to ask</span>
          </span>
          <span className="text-[#A59787]">Grounding verified by FAISS vector retriever</span>
        </div>
      </div>
    </div>
  );
};
