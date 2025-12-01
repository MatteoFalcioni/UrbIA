/**
 * MessageInput: text input for sending messages with SSE streaming support.
 * Handles user input, sends to backend, streams assistant response, and updates UI.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { Send } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { InterruptModal } from './InterruptModal';
import { ModelSelector } from './ModelSelector';
import { ContextWindowSlider } from './ContextWindowSlider';
import { CreativitySlider } from './CreativitySlider';
import { useSSE } from '@/hooks/useSSE';
import { createThread, updateThreadConfig, listMessages, getThreadState } from '@/utils/api';
import type { Message } from '@/types/api';

export function MessageInput() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const setCurrentThreadId = useChatStore((state) => state.setCurrentThreadId);
  const addMessage = useChatStore((state) => state.addMessage);
  const setMessages = useChatStore((state) => state.setMessages);
  const addThread = useChatStore((state) => state.addThread);
  const userId = useChatStore((state) => state.userId);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  const setDraft = useChatStore((state) => state.setStreamingDraft);
  const clearDraft = useChatStore((state) => state.clearStreamingDraft);
  const setThinkingBlock = useChatStore((state) => state.setThinkingBlock);
  const clearThinkingBlock = useChatStore((state) => state.clearThinkingBlock);
  const setThreads = useChatStore((state) => state.setThreads);
  const setIsSummarizing = useChatStore((state) => state.setIsSummarizing);
  const setIsReviewing = useChatStore((state) => state.setIsReviewing);
  const setTodos = useChatStore((state) => state.setTodos);
  const setCurrentReport = useChatStore((state) => state.setCurrentReport);
  const apiKeys = useChatStore((state) => state.apiKeys);
  
  const [input, setInput] = useState('');
  const [interruptData, setInterruptData] = useState<any>(null); // Track graph interrupts (HITL)
  const streamingRef = useRef(''); // Accumulate streaming tokens (mirror to avoid stale closure on onDone)
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const addToolDraft = useChatStore((state) => state.addToolDraft);
  const removeToolDraft = useChatStore((state) => state.removeToolDraft);
  const clearToolDrafts = useChatStore((state) => state.clearToolDrafts);
  const addSubagentDraft = useChatStore((state) => state.addSubagentDraft);
  const clearSubagentDrafts = useChatStore((state) => state.clearSubagentDrafts);
  const finalizeSubagentDraft = useChatStore((state) => state.finalizeSubagentDraft);
  const clearSubagentSegments = useChatStore((state) => state.clearSubagentSegments);
  const addArtifactBubble = useChatStore((state) => state.addArtifactBubble);
  const removeSubagentSegments = useChatStore((state) => state.removeSubagentSegments);
  
  // Track the currently active agent (last agent that received tokens)
  const activeAgentRef = useRef<string | null>(null);
  const clearArtifactBubbles = useChatStore((state) => state.clearArtifactBubbles);
  const setContextUsage = useChatStore((state) => state.setContextUsage);
  
  // Check if user has any API keys configured
  const hasApiKeys = Boolean(apiKeys.openai || apiKeys.anthropic);

  // Update thread state when thread changes
  useEffect(() => {
    if (currentThreadId) {
      // Fetch current state, objectives, and report for the thread
      getThreadState(currentThreadId).then((state) => {
        setTodos(state.todos || []);
        
        // Load report if available
        if (state.report_content && state.report_title) {
          setCurrentReport(state.report_content, state.report_title);
        } else {
          setCurrentReport(null, null);
        }
      }).catch((err) => {
        // Ignore 404 errors when fetching thread state (new thread)
        if (err?.response?.status !== 404) {
          console.error('Failed to fetch thread state:', err);
        }
        // Fallback to default
        setTodos([]);
        setCurrentReport(null, null);
      });
    }
  }, [currentThreadId, setTodos, setCurrentReport]);

  // Auto-resize textarea
  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`; // Max height of 200px
    }
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [input, adjustTextareaHeight]);
  
  // SSE hook with handlers for streaming events
  const { sendMessage, resumeThread, isStreaming } = useSSE({
    onThinking: (content) => {
      // Set thinking block (Claude extended thinking)
      if (currentThreadId) {
        setThinkingBlock(currentThreadId, content);
      }
    },
    onToken: (content) => {
      // Accumulate token chunks for assistant message (supervisor)
      streamingRef.current = streamingRef.current + content;
      if (currentThreadId) {
        setDraft(currentThreadId, streamingRef.current);
        // Clear all tool drafts when assistant starts responding
        clearToolDrafts(currentThreadId);
      }
    },
    onSubagentToken: (agent, content) => {
      // Accumulate token chunks for subagent messages
      if (currentThreadId) {
        // Track the active agent
        activeAgentRef.current = agent;
        
        // Get existing content or start fresh
        const existing = useChatStore.getState().subagentDrafts.find(
          (s) => s.threadId === currentThreadId && s.agent === agent
        );
        const newContent = existing ? existing.content + content : content;
        addSubagentDraft(currentThreadId, agent, newContent);
      }
    },
    onToolStart: (name, input) => {
      console.log(`Tool started: ${name}`, input);
      if (currentThreadId) {
        // Finalize the current agent's draft if there is one
        if (activeAgentRef.current) {
          finalizeSubagentDraft(currentThreadId, activeAgentRef.current);
          activeAgentRef.current = null;
        }
        addToolDraft(currentThreadId, name, input);
      }
    },
    onToolEnd: (name, output, artifacts) => {
      console.log(`Tool finished: ${name}`, output, artifacts);
      if (currentThreadId) {
        // Add a longer minimum display time so users can see all tools
        setTimeout(() => {
          removeToolDraft(currentThreadId, name);
        }, 1000); // 1 second minimum display time
        
        // Show artifacts immediately during streaming
        if (artifacts && artifacts.length > 0) {
          addArtifactBubble(currentThreadId, name, artifacts);
        }
      }
    },
    onTitleUpdated: (title) => {
      // Update thread title in sidebar when auto-title completes
      if (currentThreadId) {
        // Get current threads from store to avoid stale closure
        const currentThreads = useChatStore.getState().threads;
        // Update immediately for faster response
        setThreads(currentThreads.map((t) => (t.id === currentThreadId ? { ...t, title } : t)));
      }
    },
    onSummarizing: (status) => {
      // Show/hide summarization animation
      setIsSummarizing(status === 'start');
    },
    onReviewing: (status) => {
      // Show/hide reviewing animation
      setIsReviewing(status === 'start');
    },
    onTodosUpdated: (todos) => {
      setTodos(todos);
    },
    onReportWritten: (title, content) => {
      // Display report in artifacts panel in real-time
      setCurrentReport(content, title);
      // Auto-open artifacts panel if not already open
      const isArtifactsPanelOpen = useChatStore.getState().isArtifactsPanelOpen;
      if (!isArtifactsPanelOpen) {
        useChatStore.getState().toggleArtifactsPanel();
      }
    },
    onInterrupt: (value) => {
      // Graph interrupted - show HITL modal
      console.log('Graph interrupted:', value);
      setInterruptData(value);
    },
    onDone: async (messageId) => {
      console.log('onDone called with messageId:', messageId);
      // Reset streaming state
      streamingRef.current = '';
      clearDraft();
      clearThinkingBlock();
      
      // Finalize any remaining subagent drafts before clearing
      if (currentThreadId && activeAgentRef.current) {
        finalizeSubagentDraft(currentThreadId, activeAgentRef.current);
      }
      activeAgentRef.current = null; // Reset active agent
      
      // Clear any remaining tool drafts (handles failed tools that didn't send tool_end)
      if (currentThreadId) {
        clearToolDrafts(currentThreadId);
        clearSubagentDrafts(currentThreadId);
        // Don't clear segments here - we'll keep them visible since backend only saves one message per agent
      }
      
      // Refetch all messages from DB to get the correct order (supervisor + subagents)
      // Backend saves all messages (supervisor + subagents) so we need to refetch to see them all
      if (currentThreadId) {
        // Short delay to ensure backend has committed all messages to DB
        setTimeout(async () => {
          try {
            const fetchedMessages = await listMessages(currentThreadId);
            const chronologicalMessages = [...fetchedMessages].reverse();
            
            // Update store with fresh messages from DB (chronological order)
            setMessages(chronologicalMessages);
            
            // Remove stale frontend segments for agents that now have saved segments
            const segmentAgents = Array.from(
              new Set(
                chronologicalMessages
                  .filter(
                    (m) =>
                      m.role === 'assistant' &&
                      m.meta?.agent &&
                      m.meta?.segment_index !== undefined
                  )
                  .map((m) => m.meta!.agent!)
              )
            );
            if (segmentAgents.length > 0) {
              removeSubagentSegments(currentThreadId, segmentAgents);
            }
            
            // Check if ANY message in the thread has artifacts
            const allArtifacts = chronologicalMessages
              .filter(m => m.artifacts && m.artifacts.length > 0)
              .flatMap(m => m.artifacts || []);
            
            if (allArtifacts.length > 0) {
              // Artifacts are now in store - safe to clear bubbles
              // The de-duplicator in ArtifactDisplay will show them from messages
              clearArtifactBubbles(currentThreadId);
            } else {
              console.log('No artifacts in DB yet, keeping bubbles visible');
            }
            
            // Note: We keep subagentSegments visible because the backend only saves
            // one aggregated message per agent, but we want to show all the segments
            // that were separated by tool calls. The segments will be cleared when
            // the user switches threads or when a new stream starts.
          } catch (err: any) {
            // Ignore 404 errors (thread might have been deleted)
            if (err?.response?.status !== 404 && err?.status !== 404) {
              console.error('Failed to fetch messages after done:', err);
            }
          }
        }, 300);  // Slightly longer delay to ensure all messages (including subagents) are committed
      }
    },
    onError: (error) => {
      console.error('Stream error:', error);
      alert(`Error: ${error}`);
      streamingRef.current = '';
      clearDraft();
      clearThinkingBlock();
      
      // Clear any hanging tool drafts when stream errors out
      if (currentThreadId) {
        clearToolDrafts(currentThreadId);
      }
    },
  });

  /**
   * Send message on Enter (Shift+Enter for new line).
   */
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    if (!input.trim() || isStreaming) return;

    const userText = input.trim();
    setInput('');

    // If no thread selected, create one on the fly and select it
    let threadId = currentThreadId;
    if (!threadId) {
      try {
        const newThread = await createThread(userId, 'New chat');
        addThread(newThread);
        setCurrentThreadId(newThread.id);
        threadId = newThread.id;
        // Initialize context usage for new thread
        setContextUsage(0, defaultConfig.context_window ?? 64000);

        // Apply default config to new thread if any config is set
        if (defaultConfig.model || defaultConfig.temperature !== null || defaultConfig.system_prompt || defaultConfig.context_window !== null) {
          await updateThreadConfig(threadId, defaultConfig);
        }
      } catch (err) {
        alert('Failed to create thread');
        return;
      }
    }

    // Clear any finalized subagent segments from previous turns so they don't linger
    if (threadId) {
      clearSubagentSegments(threadId);
    }
    
    // Add user message to UI immediately (optimistic)
    const userMessageId = crypto.randomUUID();
    const userMsg: Message = {
      id: userMessageId,
      thread_id: threadId!,
      message_id: userMessageId, // Set message_id so segments can be linked to this user message
      role: 'user',
      content: { text: userText },
    };
    addMessage(userMsg);

    // Send to backend and stream response
    await sendMessage(threadId!, userMessageId, { text: userText });
  }

  return (
    <form onSubmit={handleSubmit} className="border-t p-4" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--bg-primary)' }}>
      <div className="flex gap-3 items-end">
        {/* Text input with context indicator */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              // Submit on Enter (not Shift+Enter)
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e as any);
              }
            }}
            placeholder={
              !hasApiKeys 
                ? "Please add API keys in Settings to start chatting" 
                : currentThreadId 
                  ? "Type a message..." 
                  : "Create a new thread to start chatting"
            }
            rows={1}
            disabled={isStreaming || !hasApiKeys}
            className="w-full px-4 py-3 pr-16 pb-14 rounded-xl focus:outline-none focus:ring-2 focus:border-transparent resize-none disabled:opacity-50 transition-all duration-200 text-sm overflow-hidden"
            style={{ 
              border: '1px solid var(--border)', 
              backgroundColor: 'var(--bg-secondary)', 
              color: 'var(--text-primary)',
              minHeight: '48px',
              maxHeight: '200px'
            } as React.CSSProperties}
          />

          {/* Model selector in bottom-left corner inside textarea */}
          <div className="absolute bottom-2 left-3">
            <ModelSelector />
          </div>

          {/* Sliders in bottom-right corner inside textarea */}
          <div className="absolute bottom-3 right-2 flex flex-col gap-1">
            <ContextWindowSlider />
            <CreativitySlider />
          </div>
        </div>

        {/* Send button (hidden when streaming) */}
        {!isStreaming && (
          <button
            type="submit"
            disabled={!input.trim() || !currentThreadId || !hasApiKeys}
            className="px-4 py-3 rounded-xl transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md disabled:shadow-none disabled:cursor-not-allowed"
            style={{ 
              backgroundColor: 'var(--user-message-bg)', 
              color: 'var(--user-message-text)',
              opacity: (!input.trim() || !currentThreadId || !hasApiKeys) ? 0.5 : 1
            }}
          >
            <Send size={18} />
          </button>
        )}

      </div>

      {/* Inline typing now handled in MessageList; no bottom preview */}

      {/* Active tool indicators are now rendered inline in the chat list */}
      
      {/* Interrupt Modal - Show when graph is interrupted for HITL */}
      {interruptData && currentThreadId && (
        <InterruptModal
          interruptData={interruptData}
          onResume={(resumeValue) => {
            resumeThread(currentThreadId, resumeValue);
            setInterruptData(null);
          }}
          onCancel={() => setInterruptData(null)}
        />
      )}
    </form>
  );
}

