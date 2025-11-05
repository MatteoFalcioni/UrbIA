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
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: Record<string, any> | null;
  tool_name?: string | null;
  tool_input?: Record<string, any> | null;
  tool_output?: Record<string, any> | null;
  artifacts?: Artifact[];
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

// SSE event types from backend streaming
export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'thinking'; content: string }
  | { type: 'tool_start'; name: string; input: any }
  | { type: 'tool_end'; name: string; output: any; artifacts?: Artifact[] }
  | { type: 'title_updated'; title: string }
  | { type: 'context_update'; tokens_used: number; max_tokens: number }
  | { type: 'summarizing'; status: 'start' | 'done' }
  | { type: 'interrupt'; value: any }  // Graph interrupt for human-in-the-loop
  | { type: 'done'; message_id: string | null }
  | { type: 'error'; error: string };

