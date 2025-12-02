/**
 * useSSE: hook for Server-Sent Events (SSE) streaming from POST /messages endpoint.
 * Handles EventSource connection, token streaming, tool events, and completion.
 */

import { useCallback, useRef, useState } from 'react';
import type { SSEEvent, Todo } from '@/types/api';

import type { Artifact } from '@/types/api';

const BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface UseSSEOptions {
  onToken?: (content: string) => void;
  onThinking?: (content: string) => void;
  onSubagentToken?: (agent: string, content: string) => void;  // Streaming from subagents
  onToolStart?: (name: string, input: any) => void;
  onToolEnd?: (name: string, output: any, artifacts?: Artifact[]) => void;
  onTitleUpdated?: (title: string) => void;
  onSummarizing?: (status: 'start' | 'done') => void;
  onReviewing?: (status: 'start' | 'done') => void;
  onTodosUpdated?: (todos: Todo[]) => void;
  onReportWritten?: (title: string, content: string) => void;
  onInterrupt?: (value: any) => void;  // Called when graph is interrupted (HITL)
  onDone?: (messageId: string | null) => void;
  onError?: (error: string) => void;
}

export function useSSE(options: UseSSEOptions) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [canContinue, setCanContinue] = useState(false); // Track if we can continue after stop
  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * Send a message and stream the assistant response via SSE.
   * 
   * @param threadId - Target thread ID
   * @param messageId - Client-generated UUID for idempotency
   * @param content - Message content (should have {text: string})
   */
  const sendMessage = useCallback(
    async (threadId: string, messageId: string, content: Record<string, any>) => {
      // Abort any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new AbortController for this request
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      setIsStreaming(true);
      setCanContinue(false); // Reset continue flag when starting new message

      try {
        // POST message with fetch to initiate SSE stream
        const res = await fetch(`${BASE_URL}/threads/${threadId}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message_id: messageId,
            content,
            role: 'user',
          }),
          signal: abortController.signal, // Add abort signal
        });

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Failed to send message: ${res.status} ${errorText}`);
        }

        // Read SSE stream from response body
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('No response body');
        }

        // Process SSE stream
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6); // Remove 'data: ' prefix
              try {
                const event: SSEEvent = JSON.parse(data);

                // Dispatch to handlers based on event type
                if (event.type === 'token') {
                  options.onToken?.(event.content);
                } else if (event.type === 'thinking') {
                  options.onThinking?.(event.content);
                } else if (event.type === 'subagent_token') {
                  options.onSubagentToken?.(event.agent, event.content);
                } else if (event.type === 'tool_start') {
                  options.onToolStart?.(event.name, event.input);
                } else if (event.type === 'tool_end') {
                  options.onToolEnd?.(event.name, event.output, event.artifacts);
                } else if (event.type === 'title_updated') {
                  options.onTitleUpdated?.(event.title);
                } else if (event.type === 'summarizing') {
                  options.onSummarizing?.(event.status);
                } else if (event.type === 'reviewing') {
                  options.onReviewing?.(event.status);
                } else if (event.type === 'todos_updated') {
                  options.onTodosUpdated?.(event.todos);
                } else if (event.type === 'report_written') {
                  options.onReportWritten?.(event.title, event.content);
                } else if (event.type === 'interrupt') {
                  options.onInterrupt?.(event.value);
                  setIsStreaming(false);  // Stream ends after interrupt
                  setCanContinue(false); // Can't continue after interrupt - need resume with decision
                } else if (event.type === 'done') {
                  options.onDone?.(event.message_id);
                  setIsStreaming(false);
                  setCanContinue(false); // Execution complete
                } else if (event.type === 'error') {
                  options.onError?.(event.error);
                  setIsStreaming(false);
                  setCanContinue(false);
                }
              } catch (parseErr) {
                console.warn('Failed to parse SSE event:', data, parseErr);
              }
            }
          }
        }
      } catch (err) {
        // Don't treat abort as an error
        if (err instanceof Error && err.name === 'AbortError') {
          console.log('Stream aborted by user');
          setCanContinue(true); // Enable continue button after user stop
        } else {
          options.onError?.(err instanceof Error ? err.message : 'Stream failed');
          setCanContinue(false);
        }
        setIsStreaming(false);
      } finally {
        // Clean up abort controller reference
        if (abortControllerRef.current === abortController) {
          abortControllerRef.current = null;
        }
      }
    },
    [options]
  );

  /**
   * Resume a thread after an interrupt.
   * 
   * @param threadId - Target thread ID
   * @param resumeValue - Resume data (e.g., {type: 'accept'}, {type: 'edit', edit_instructions: '...'})
   */
  const resumeThread = useCallback(
    async (threadId: string, resumeValue: Record<string, any>) => {
      // Abort any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      setIsStreaming(true);
      setCanContinue(false); // Reset continue flag when resuming

      try {
        // POST resume request
        const res = await fetch(`${BASE_URL}/threads/${threadId}/resume`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ resume_value: resumeValue }),
          signal: abortController.signal,
        });

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(`Failed to resume: ${res.status} ${errorText}`);
        }

        // Read SSE stream (same as sendMessage)
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('No response body');
        }

        // Process SSE stream
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              try {
                const event: SSEEvent = JSON.parse(data);

                // Dispatch to handlers (same as sendMessage)
                if (event.type === 'token') {
                  options.onToken?.(event.content);
                } else if (event.type === 'thinking') {
                  options.onThinking?.(event.content);
                } else if (event.type === 'subagent_token') {
                  options.onSubagentToken?.(event.agent, event.content);
                } else if (event.type === 'tool_start') {
                  options.onToolStart?.(event.name, event.input);
                } else if (event.type === 'tool_end') {
                  options.onToolEnd?.(event.name, event.output, event.artifacts);
                } else if (event.type === 'title_updated') {
                  options.onTitleUpdated?.(event.title);
                } else if (event.type === 'summarizing') {
                  options.onSummarizing?.(event.status);
                } else if (event.type === 'reviewing') {
                  options.onReviewing?.(event.status);
                } else if (event.type === 'todos_updated') {
                  options.onTodosUpdated?.(event.todos);
                } else if (event.type === 'report_written') {
                  options.onReportWritten?.(event.title, event.content);
                } else if (event.type === 'interrupt') {
                  options.onInterrupt?.(event.value);
                  setIsStreaming(false);
                  setCanContinue(false); // Can't continue after interrupt - need resume with decision
                } else if (event.type === 'done') {
                  options.onDone?.(event.message_id);
                  setIsStreaming(false);
                  setCanContinue(false); // Execution complete
                } else if (event.type === 'error') {
                  options.onError?.(event.error);
                  setIsStreaming(false);
                  setCanContinue(false);
                }
              } catch (parseErr) {
                console.warn('Failed to parse SSE event:', data, parseErr);
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          console.log('Resume stream aborted');
          setCanContinue(true); // Enable continue button after user stop
        } else {
          options.onError?.(err instanceof Error ? err.message : 'Resume failed');
          setCanContinue(false);
        }
        setIsStreaming(false);
      } finally {
        if (abortControllerRef.current === abortController) {
          abortControllerRef.current = null;
        }
      }
    },
    [options]
  );

  /**
   * Stop the current stream (user-initiated).
   * Sets canContinue flag so the continue button can be shown.
   */
  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreaming(false);
      setCanContinue(true); // Enable continue button
      console.log('Stream stopped by user');
    }
  }, []);

  /**
   * Continue execution from the last checkpoint after user stopped.
   * 
   * @param threadId - Target thread ID
   */
  const continueThread = useCallback(
    async (threadId: string) => {
      // Abort any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      setIsStreaming(true);
      setCanContinue(false); // Hide continue button while streaming

      try {
        // POST continue request (no body needed)
        const res = await fetch(`${BASE_URL}/threads/${threadId}/continue`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: abortController.signal,
        });

        if (!res.ok) {
          const errorText = await res.text();
          // Special handling for "no execution to continue" error
          if (res.status === 400 && errorText.includes('No execution to continue')) {
            options.onError?.('This conversation has already completed. Nothing to continue.');
            setIsStreaming(false);
            setCanContinue(false);
            return;
          }
          throw new Error(`Failed to continue: ${res.status} ${errorText}`);
        }

        // Read SSE stream (same as sendMessage/resumeThread)
        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('No response body');
        }

        // Process SSE stream
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              try {
                const event: SSEEvent = JSON.parse(data);

                // Dispatch to handlers (same as sendMessage)
                if (event.type === 'token') {
                  options.onToken?.(event.content);
                } else if (event.type === 'thinking') {
                  options.onThinking?.(event.content);
                } else if (event.type === 'subagent_token') {
                  options.onSubagentToken?.(event.agent, event.content);
                } else if (event.type === 'tool_start') {
                  options.onToolStart?.(event.name, event.input);
                } else if (event.type === 'tool_end') {
                  options.onToolEnd?.(event.name, event.output, event.artifacts);
                } else if (event.type === 'title_updated') {
                  options.onTitleUpdated?.(event.title);
                } else if (event.type === 'summarizing') {
                  options.onSummarizing?.(event.status);
                } else if (event.type === 'reviewing') {
                  options.onReviewing?.(event.status);
                } else if (event.type === 'todos_updated') {
                  options.onTodosUpdated?.(event.todos);
                } else if (event.type === 'report_written') {
                  options.onReportWritten?.(event.title, event.content);
                } else if (event.type === 'interrupt') {
                  options.onInterrupt?.(event.value);
                  setIsStreaming(false);
                  setCanContinue(false); // Can't continue after interrupt - need to resume with decision
                } else if (event.type === 'done') {
                  options.onDone?.(event.message_id);
                  setIsStreaming(false);
                  setCanContinue(false); // Execution complete
                } else if (event.type === 'error') {
                  options.onError?.(event.error);
                  setIsStreaming(false);
                  setCanContinue(false);
                }
              } catch (parseErr) {
                console.warn('Failed to parse SSE event:', data, parseErr);
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          console.log('Continue stream aborted');
          setCanContinue(true); // Can continue again after abort
        } else {
          options.onError?.(err instanceof Error ? err.message : 'Continue failed');
          setCanContinue(false);
        }
        setIsStreaming(false);
      } finally {
        if (abortControllerRef.current === abortController) {
          abortControllerRef.current = null;
        }
      }
    },
    [options]
  );

  return { sendMessage, resumeThread, stopStream, continueThread, isStreaming, canContinue };
}

