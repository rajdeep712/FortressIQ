import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, X, FileText, Loader2 } from 'lucide-react';
import { useViewerStore } from '../store/useViewerStore';
import { RAGDocument } from '../types';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose }) => {
  const [dragActive, setDragActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [fileName, setFileName] = useState('');
  const uploadDocument = useViewerStore((s) => s.uploadDocument);

  if (!isOpen) return null;

  const handleSimulatedUpload = (name: string) => {
    setFileName(name);
    setIsProcessing(true);
    setProcessingStep(1); // Parsing PDF

    setTimeout(() => {
      setProcessingStep(2); // Splitting into 512-token chunks
    }, 600);

    setTimeout(() => {
      setProcessingStep(3); // Generating Dense DPR Embeddings
    }, 1200);

    setTimeout(() => {
      setProcessingStep(4); // Building FAISS HNSW index
    }, 1800);

    setTimeout(() => {
      const newDoc: RAGDocument = {
        id: `doc-${Date.now()}`,
        title: name.replace(/\.[^/.]+$/, ''),
        filename: name,
        totalPages: 3,
        fileType: 'pdf',
        uploadedAt: 'Just now',
        fileSize: '1.9 MB',
        category: 'Custom Upload',
        authors: 'User Uploaded Research',
        abstract: 'User indexed technical PDF document with 12 extracted passages and DPR embedding representations.',
        pages: [
          {
            pageNumber: 1,
            title: '1. Document Overview & Executive Summary',
            sectionHeader: 'USER UPLOADED KNOWLEDGE BASE',
            content: [
              {
                type: 'heading',
                text: name.replace(/\.[^/.]+$/, '')
              },
              {
                type: 'callout',
                text: 'This document was indexed and partitioned into dense vector chunks with DPR bi-encoders.'
              },
              {
                type: 'paragraph',
                isChunk: true,
                citationId: 9,
                chunkMeta: {
                  id: 'chunk-user-doc-1',
                  confidence: 0.96,
                  label: 'Core Synthesis & Primary Thesis'
                },
                text: 'The architecture incorporates modern sparse and dense memory retrieval pipelines, allowing accurate contextual synthesis with zero factual drift.'
              }
            ]
          },
          {
            pageNumber: 2,
            title: '2. Implementation Details & Technical Benchmarks',
            sectionHeader: 'SECTION 2 • EVALUATION & EMBEDDING METRICS',
            content: [
              {
                type: 'heading',
                text: '2. Experimental Results & Latency'
              },
              {
                type: 'paragraph',
                text: 'Across extensive empirical testing, the vector chunk indexing latency maintained sub-10ms query times over 100,000 embedded documents.'
              }
            ]
          },
          {
            pageNumber: 3,
            title: '3. Conclusion & Recommendations',
            sectionHeader: 'SECTION 3 • CONCLUSION',
            content: [
              {
                type: 'heading',
                text: '3. Summary'
              },
              {
                type: 'paragraph',
                text: 'In conclusion, the retrieved chunks demonstrate high semantic correlation with user research queries.'
              }
            ]
          }
        ],
        chunks: [
          {
            id: 'chunk-user-doc-1',
            citationId: 9,
            paragraphIndex: 1,
            chunkTitle: 'Core Synthesis & Primary Thesis',
            confidence: 0.96,
            text: 'The architecture incorporates modern sparse and dense memory retrieval pipelines, allowing accurate contextual synthesis with zero factual drift.',
            boundingBox: {
              top: 32,
              left: 6,
              width: 88,
              height: 18
            }
          }
        ]
      };

      uploadDocument(newDoc);
      setIsProcessing(false);
      onClose();
    }, 2400);
  };

  const steps = [
    'Parsing PDF text streams & font tables...',
    'Splitting text into 512-token overlapping chunks...',
    'Generating Dense Passage Retrieval (DPR) 768-d embeddings...',
    'Inserting vectors into FAISS HNSW index...'
  ];

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
          onClick={onClose}
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
              Index New Research Paper
            </h3>
            <p className="text-xs text-[#7A6D5E]">
              Add PDFs or documents to the RAG FAISS vector store
            </p>
          </div>
        </div>

        {isProcessing ? (
          <div className="py-8 text-center space-y-4">
            <Loader2 className="w-10 h-10 text-[#F97316] animate-spin mx-auto" />
            <div>
              <h4 className="text-sm font-bold text-[#221C16] mb-1">
                Vectorizing {fileName}...
              </h4>
              <p className="text-xs text-[#EA580C] font-mono font-medium">
                {steps[processingStep - 1] || 'Processing document...'}
              </p>
            </div>

            <div className="w-full bg-[#EFE9DF] rounded-full h-2 overflow-hidden mt-4">
              <motion.div
                className="bg-gradient-to-r from-[#FF7A00] to-[#EA580C] h-full"
                animate={{ width: `${processingStep * 25}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
          </div>
        ) : (
          <div>
            {/* Drag and Drop Zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                if (e.dataTransfer.files?.[0]) {
                  handleSimulatedUpload(e.dataTransfer.files[0].name);
                }
              }}
              onClick={() => {
                handleSimulatedUpload('Mixtral_8x7B_MoE_Architecture.pdf');
              }}
              className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${
                dragActive
                  ? 'border-[#F97316] bg-[#F97316]/10'
                  : 'border-[#DDD2C0] hover:border-[#F97316]/60 bg-white/60 hover:bg-white'
              }`}
            >
              <FileText className="w-10 h-10 text-[#EA580C] mx-auto mb-3" />
              <p className="text-sm font-bold text-[#221C16] mb-1">
                Click or drag & drop your PDF here
              </p>
              <p className="text-xs text-[#8A7B6B]">
                Supports PDF, DOCX, TXT up to 50MB
              </p>
            </div>

            {/* Quick Demo Pre-sets */}
            <div className="mt-4">
              <span className="text-[11px] font-bold text-[#8C7E6E] uppercase tracking-wider block mb-2">
                Or choose sample papers to index instantly:
              </span>
              <div className="space-y-1.5">
                {[
                  { name: 'Mixtral_8x7B_MoE_Architecture.pdf', size: '2.8 MB', tag: 'Sparse MoE' },
                  { name: 'FlashAttention_2_Fast_IO_Aware.pdf', size: '1.5 MB', tag: 'GPU Kernels' }
                ].map((sample) => (
                  <button
                    key={sample.name}
                    type="button"
                    onClick={() => handleSimulatedUpload(sample.name)}
                    className="w-full p-2.5 rounded-xl bg-white hover:bg-[#F6EFE5] border border-[#E5DEC3] text-left text-xs font-semibold text-[#292017] flex items-center justify-between transition-colors cursor-pointer"
                  >
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-[#EA580C]" />
                      <span>{sample.name}</span>
                    </div>
                    <span className="text-[10px] font-mono bg-[#EFE9DF] text-[#7A6C5B] px-1.5 py-0.5 rounded">
                      {sample.tag}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};
