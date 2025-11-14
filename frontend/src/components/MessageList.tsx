/**
 * MessageList: displays all messages for the current thread.
 * Fetches messages when thread changes and renders user/assistant/tool messages distinctly.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { ArrowDown } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { listMessages } from '@/utils/api';
import type { Message } from '@/types/api';
import { ThinkingBlock } from './ThinkingBlock';

export function MessageList() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const messages = useChatStore((state) => state.messages);
  const setMessages = useChatStore((state) => state.setMessages);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const streamingDraft = useChatStore((state) => state.streamingDraft);
  const thinkingBlock = useChatStore((state) => state.thinkingBlock);
  const toolDrafts = useChatStore((state) => state.toolDrafts);
  const subagentDrafts = useChatStore((state) => state.subagentDrafts);
  const subagentSegments = useChatStore((state) => state.subagentSegments);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const clearSubagentSegments = useChatStore((state) => state.clearSubagentSegments);
  const prevThreadIdRef = useRef<string | null>(null);

  // Fetch messages when thread changes
  useEffect(() => {
    // Clear segments for the previous thread when switching
    if (prevThreadIdRef.current && prevThreadIdRef.current !== currentThreadId) {
      clearSubagentSegments(prevThreadIdRef.current);
    }
    prevThreadIdRef.current = currentThreadId;

    if (!currentThreadId) {
      setMessages([]);
      return;
    }

    async function loadMessages() {
      try {
        const fetchedMessages = await listMessages(currentThreadId!, 100);
        // Messages come in desc order from API; reverse for chronological display
        setMessages(fetchedMessages.reverse());
      } catch (err) {
        console.error('Failed to load messages:', err);
      }
    }
    loadMessages();
  }, [currentThreadId, setMessages, clearSubagentSegments]);

  // Check if user is near bottom (to show/hide scroll button)
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    
    // Show button if more than 100px from bottom
    setShowScrollButton(distanceFromBottom > 100);
  }, []);

  // Scroll to bottom function
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Auto-scroll when new AI messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingDraft, thinkingBlock, toolDrafts, subagentDrafts, subagentSegments]);

  if (!currentThreadId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 dark:text-slate-400">
        Select a thread or create a new one to start chatting.
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 dark:text-slate-400">
        No messages yet. Type below to start the conversation.
      </div>
    );
  }

  return (
    <div className="relative h-full">
      <div 
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="space-y-4 p-4 h-full overflow-y-auto"
      >
        {messages
          .filter(msg => msg.role !== 'tool') // Hide tool messages from permanent display
          .map((msg) => (
            <MessageBubble 
              key={msg.id} 
              message={msg} 
            />
          ))}
        
        {/* Thinking block (Claude extended thinking) */}
        {thinkingBlock && thinkingBlock.threadId === currentThreadId && (
          <ThinkingBlock content={thinkingBlock.content} />
        )}

        {/* Finalized subagent segments (completed bubbles before tool calls) */}
        {/* Only show segments if there's no saved message for the same agent yet */}
        {subagentSegments
          .filter((s) => {
            if (s.threadId !== currentThreadId) return false;
            // Hide segment if there's a saved message for the same agent
            const hasSavedMessage = messages.some(
              (msg) => msg.role === 'assistant' && msg.meta?.agent === s.agent
            );
            return !hasSavedMessage;
          })
          .map((s) => (
            <SubagentBubble key={s.id} agent={s.agent} content={s.content} />
          ))}

        {/* Inline subagent streaming drafts */}
        {subagentDrafts
          .filter((s) => s.threadId === currentThreadId)
          .map((s, idx) => (
            <SubagentBubble key={`subagent-draft-${idx}-${s.agent}`} agent={s.agent} content={s.content} />
          ))}

        {/* Inline tool execution drafts (no artifacts here) */}
        {toolDrafts
          .filter((t) => t.threadId === currentThreadId)
          .map((t, idx) => (
            <ToolCallBubble key={`tool-draft-${idx}-${t.name}`} name={t.name} input={t.input} />
          ))}

        {/* Inline typing bubble for assistant draft */}
        {streamingDraft && streamingDraft.threadId === currentThreadId && (
          <div className="flex gap-3 items-start">
            <div className="flex-1 w-full rounded-xl p-4 shadow-sm" style={{ backgroundColor: 'var(--assistant-message-bg)', color: 'var(--assistant-message-text)' }}>
              <p className="text-sm whitespace-pre-wrap">{streamingDraft.text}</p>
            </div>
          </div>
        )}
        
        {/* Invisible anchor for scroll target */}
        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to bottom button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-4 right-4 p-3 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm border border-gray-200/50 dark:border-slate-700/50 rounded-full shadow-lg hover:shadow-xl hover:bg-white dark:hover:bg-slate-800 transition-all duration-300 ease-out flex items-center justify-center group hover:scale-110 active:scale-95"
          title="Scroll to bottom"
        >
          <ArrowDown size={18} className="text-gray-600 dark:text-slate-300 group-hover:text-gray-800 dark:group-hover:text-slate-100 transition-colors duration-200" />
        </button>
      )}
    </div>
  );
}

/**
 * ToolCallBubble: renders a collapsible tool execution indicator with shimmer animation.
 */
interface ToolCallBubbleProps {
  name: string;
  input?: any;
}

function ToolCallBubble({ name, input }: ToolCallBubbleProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div 
      className="flex gap-3 items-start cursor-pointer transition-all duration-200 hover:opacity-80"
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div 
        className="flex-1 rounded-xl px-4 py-3 shadow-sm backdrop-blur-sm border transition-all"
        style={{ 
          backgroundColor: 'color-mix(in srgb, var(--bg-secondary) 60%, transparent)',
          borderColor: 'color-mix(in srgb, var(--border) 50%, transparent)',
        }}
      >
        <div className="font-mono text-xs font-medium shimmer-text">
          {name}
        </div>
        {isExpanded && input && (
          <div 
            className="mt-2 text-xs font-mono opacity-70 pt-2 border-t"
            style={{ borderColor: 'color-mix(in srgb, var(--border) 30%, transparent)' }}
          >
            {formatParams(input)}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * SubagentBubble: renders a collapsible subagent message with dropdown (translucido style).
 */
interface SubagentBubbleProps {
  agent: string;
  content: string;
}

function SubagentBubble({ agent, content }: SubagentBubbleProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Format agent name for display
  const agentDisplayName = agent
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  return (
    <div 
      className="flex gap-3 items-start cursor-pointer transition-all duration-200 hover:opacity-80"
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div 
        className="flex-1 rounded-xl px-4 py-3 shadow-sm backdrop-blur-sm border transition-all"
        style={{ 
          backgroundColor: 'color-mix(in srgb, var(--bg-secondary) 60%, transparent)',
          borderColor: 'color-mix(in srgb, var(--border) 50%, transparent)',
        }}
      >
        <div className="font-mono text-xs font-medium shimmer-text">
          {agentDisplayName}
        </div>
        {isExpanded && content && (
          <div 
            className="mt-2 text-sm opacity-90 pt-2 border-t whitespace-pre-wrap"
            style={{ borderColor: 'color-mix(in srgb, var(--border) 30%, transparent)' }}
          >
            {content}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * MessageBubble: renders a single message with role-specific styling.
 * Supports user, assistant, and tool messages.
 */
interface MessageBubbleProps {
  message: Message;
}

// Compact parameter formatter for tool inputs/outputs
function formatParams(value: any): string {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  try {
    if (Array.isArray(value)) return value.map(String).join(', ');
    if (typeof value === 'object') {
      // Filter out tool_call_id and other internal metadata
      const filteredEntries = Object.entries(value)
        .filter(([k]) => k !== 'tool_call_id');
      
      if (filteredEntries.length === 0) return '';
      
      return filteredEntries
        .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
        .join(' · ');
    }
    return String(value);
  } catch {
    return String(value);
  }
}

function MessageBubble({ message }: MessageBubbleProps) {
  const { role, content } = message;

  // Note: Tool messages are filtered out and not displayed in permanent chat

  // User message rendering
  if (role === 'user') {
    return (
      <div className="flex gap-3 items-start justify-end">
        <div className="flex-1 w-full rounded-xl p-4 shadow-sm" style={{ backgroundColor: 'var(--user-message-bg)', color: 'var(--user-message-text)' }}>
          <p className="text-sm whitespace-pre-wrap leading-relaxed">
            {content?.text || JSON.stringify(content)}
          </p>
        </div>
      </div>
    );
  }

  // Assistant message rendering
  if (role === 'assistant') {
    // Check if this is a subagent message (has meta.agent)
    if (message.meta?.agent) {
      return <SubagentBubble agent={message.meta.agent} content={content?.text || JSON.stringify(content)} />;
    }
    
    // Regular supervisor message
    return (
      <div className="flex gap-3 items-start">
        <div className="flex-1 w-full rounded-xl p-4 shadow-sm" style={{ backgroundColor: 'var(--assistant-message-bg)', color: 'var(--assistant-message-text)' }}>
          <p className="text-sm whitespace-pre-wrap leading-relaxed">
            {content?.text || JSON.stringify(content)}
          </p>
        </div>
      </div>
    );
  }

  // Fallback for other roles
  return (
    <div className="flex gap-3 items-start">
      <div className="flex-1 bg-gray-100 dark:bg-slate-700 rounded-lg p-3">
        <p className="text-xs text-gray-500 dark:text-slate-400 mb-1">
          {role}
        </p>
        <p className="text-sm whitespace-pre-wrap">
          {content?.text || JSON.stringify(content)}
        </p>
      </div>
    </div>
  );
}

