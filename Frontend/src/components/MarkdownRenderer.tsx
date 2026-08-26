import React from 'react';
import Markdown from 'react-markdown';
import { Citation } from '../types';
import { CitationBadge } from './CitationBadge';

interface MarkdownRendererProps {
  content: string;
  citations?: Citation[];
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, citations = [] }) => {
  // Build a lookup map for quick citation retrieval by id
  const citationMap = React.useMemo(() => {
    const map = new Map<number, Citation>();
    citations.forEach((c) => map.set(c.id, c));
    return map;
  }, [citations]);

  // Helper to replace [1], [2], etc. inside text nodes with interactive CitationBadge components
  const renderTextWithCitations = (text: string) => {
    if (!text || typeof text !== 'string') return text;

    // Pattern matches citations like [1], [2], [1, 2]
    const regex = /\[(\d+)\]/g;
    const parts: (string | React.ReactNode)[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(text)) !== null) {
      // Add preceding plain text
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      const citationNum = parseInt(match[1], 10);
      const citation = citationMap.get(citationNum);

      if (citation) {
        parts.push(<CitationBadge key={`cite-${citation.id}-${match.index}`} citation={citation} />);
      } else {
        // If not in this message's citations list, create a basic fallback badge or render text
        parts.push(
          <span
            key={`unknown-cite-${citationNum}-${match.index}`}
            className="inline-flex items-center px-1.5 py-0.5 text-xs font-mono font-bold rounded-md bg-[#F97316]/10 text-[#EA580C] border border-[#F97316]/20 mx-0.5"
          >
            [{citationNum}]
          </span>
        );
      }

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  return (
    <div className="prose prose-stone max-w-none text-[#2D2620] text-sm leading-relaxed">
      <Markdown
        components={{
          p: ({ children }) => {
            const processedChildren = React.Children.map(children, (child) => {
              if (typeof child === 'string') {
                return renderTextWithCitations(child);
              }
              return child;
            });
            return <p className="mb-3 last:mb-0 leading-relaxed text-[#352B23]">{processedChildren}</p>;
          },
          li: ({ children }) => {
            const processedChildren = React.Children.map(children, (child) => {
              if (typeof child === 'string') {
                return renderTextWithCitations(child);
              }
              return child;
            });
            return <li className="mb-1.5 text-[#352B23]">{processedChildren}</li>;
          },
          h3: ({ children }) => (
            <h3 className="text-base font-bold text-[#1F1914] mt-3.5 mb-2 font-display tracking-wide">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-sm font-semibold text-[#2D241C] mt-3 mb-1.5">{children}</h4>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-[#18130E]">{children}</strong>
          ),
          code: ({ children }) => (
            <code className="px-1.5 py-0.5 rounded bg-[#EFE9DF] text-[#7C2D12] text-xs font-mono border border-[#E2D8CA]">
              {children}
            </code>
          ),
          ul: ({ children }) => (
            <ul className="list-disc pl-5 my-2 space-y-1">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 my-2 space-y-1">{children}</ol>
          )
        }}
      >
        {content}
      </Markdown>
    </div>
  );
};
