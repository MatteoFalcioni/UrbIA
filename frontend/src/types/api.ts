/**
 * TypeScript types matching backend API schemas.
 * Keep in sync with backend/app/api.py Pydantic models.
 */

export interface Thread {
  id: string;
  user_id: string;
  title: string | null;
  archived_at?: string | null;
}

export interface Message {
  id: string;
  thread_id: string;
  message_id?: string | null;  // Client-supplied idempotency key, also used to link segments to user messages
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: Record<string, any> | null;
  tool_name?: string | null;
  tool_input?: Record<string, any> | null;
  tool_output?: Record<string, any> | null;
  meta?: { agent?: string; segment_index?: number } | null;  // For subagent messages: { agent: 'data_analyst' | 'report_writer' | 'reviewer', segment_index?: number }
  artifacts?: Artifact[];
  created_at?: string;  // ISO timestamp from backend
}

export interface ThreadConfig {
  model: string | null;
  temperature: number | null;
  system_prompt: string | null;
  context_window: number | null;
  settings: Record<string, any> | null;
}

// Artifact metadata from code execution
export interface Artifact {
  id: string;
  name: string;
  mime: string;
  size: number;
  url: string;
  sha256?: string;
  created_at?: string;
}

export interface Todo {
  content: string;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
}

// SSE event types from backend streaming
export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'thinking'; content: string }
  | { type: 'subagent_token'; agent: string; content: string }  // Streaming from subagents (data_analyst, report_writer, reviewer)
  | { type: 'tool_start'; name: string; input: any }
  | { type: 'tool_end'; name: string; output: any; artifacts?: Artifact[] }
  | { type: 'title_updated'; title: string }
  | { type: 'summarizing'; status: 'start' | 'done' }
  | { type: 'reviewing'; status: 'start' | 'done' }
  | { type: 'todos_updated'; todos: Todo[] }
  | { type: 'report_written'; title: string; content: string }
  | { type: 'score_updated'; score: number }  // Reviewer final score (0-1)
  | { type: 'interrupt'; value: any }  // Graph interrupt for human-in-the-loop
  | { type: 'done'; message_id: string | null }
  | { type: 'error'; error: string };

