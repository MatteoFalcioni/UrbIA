/**
 * Global state management using Zustand.
 * Handles threads, messages, and UI state.
 */

import { create } from 'zustand';
import type { Thread, Message, Artifact, Todo } from '@/types/api';
import type { ToastType } from '@/components/Toast';

interface ChatStore {
  // User identity (anonymous UUID stored in localStorage)
  userId: string;
  setUserId: (id: string) => void;

  // Threads
  threads: Thread[];
  setThreads: (threads: Thread[]) => void;
  addThread: (thread: Thread) => void;
  updateThread: (threadId: string, updates: Partial<Thread>) => void;
  removeThread: (threadId: string) => void;

  // Current selected thread
  currentThreadId: string | null;
  setCurrentThreadId: (id: string | null) => void;

  // Messages for current thread
  messages: Message[];
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;

  // Streaming draft for inline assistant typing bubble
  streamingDraft: { threadId: string; text: string } | null;
  setStreamingDraft: (threadId: string, text: string) => void;
  clearStreamingDraft: () => void;

  // Thinking block (Claude extended thinking before tool calls)
  thinkingBlock: { threadId: string; content: string } | null;
  setThinkingBlock: (threadId: string, content: string) => void;
  clearThinkingBlock: () => void;

  // Live tool execution drafts shown inline while tools run
  toolDrafts: { threadId: string; name: string; input?: any }[];
  addToolDraft: (threadId: string, name: string, input?: any) => void;
  removeToolDraft: (threadId: string, name: string) => void;
  clearToolDrafts: (threadId: string) => void;
  
  // Subagent streaming drafts (data_analyst, report_writer, reviewer)
  subagentDrafts: { threadId: string; agent: string; content: string }[];
  addSubagentDraft: (threadId: string, agent: string, content: string) => void;
  updateSubagentDraft: (threadId: string, agent: string, content: string) => void;
  clearSubagentDrafts: (threadId: string) => void;
  finalizeSubagentDraft: (threadId: string, agent: string) => void;
  
  // Subagent segments (finalized bubbles that persist until backend saves)
  subagentSegments: { threadId: string; agent: string; content: string; id: string; timestamp: number }[];
  clearSubagentSegments: (threadId: string) => void;
  removeSubagentSegments: (threadId: string, agents: string[]) => void;
  
  // Artifact bubbles (separate from tool drafts, persistent)
  artifactBubbles: { threadId: string; toolName: string; artifacts: Artifact[] }[];
  addArtifactBubble: (threadId: string, toolName: string, artifacts: Artifact[]) => void;
  clearArtifactBubbles: (threadId: string) => void;

  // UI state
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  theme: 'light' | 'dark' | 'auto';
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;

  // Sidebar width (resizable with limits)
  sidebarWidth: number;
  setSidebarWidth: (px: number) => void;

  // Right sidebar (artifacts panel)
  isArtifactsPanelOpen: boolean;
  toggleArtifactsPanel: () => void;
  artifactsPanelWidth: number;
  setArtifactsPanelWidth: (px: number) => void;
  reports: { title: string; content: string }[];
  setReports: (reports: { title: string; content: string }[]) => void;
  currentReport: string | null;
  currentReportTitle: string | null;
  setCurrentReport: (report: string | null, title?: string | null) => void;
  todos: Todo[];
  setTodos: (todos: Todo[]) => void;
  
  // Analysis quality score (from reviewer)
  analysisScore: number | null;
  analysisStatus: 'pending' | 'approved' | 'rejected' | 'limit_exceeded' | 'end_flow' | null;
  setAnalysisScore: (score: number | null) => void;
  setAnalysisStatus: (status: ChatStore['analysisStatus']) => void;

  // Default configs for new threads (applied when auto-creating)
  defaultConfig: { model: string | null; temperature: number | null; system_prompt: string | null; context_window: number | null };
  setDefaultConfig: (config: { model?: string | null; temperature?: number | null; system_prompt?: string | null; context_window?: number | null }) => void;

  // Context usage tracking
  contextUsage: { tokensUsed: number; maxTokens: number };
  setContextUsage: (tokensUsed: number, maxTokens: number) => void;
  isSummarizing: boolean;
  setIsSummarizing: (value: boolean) => void;
  isReviewing: boolean;
  setIsReviewing: (value: boolean) => void;

  // Toast notifications
  toasts: Array<{ id: string; type: ToastType; title: string; message?: string; duration?: number }>;
  addToast: (type: ToastType, title: string, message?: string, duration?: number) => void;
  removeToast: (id: string) => void;

  // Bulk selection for threads
  selectedThreadIds: Set<string>;
  toggleThreadSelection: (threadId: string) => void;
  selectAllThreads: (threadIds: string[]) => void;
  clearThreadSelection: () => void;

  // API Keys
  apiKeys: { openai: string | null; anthropic: string | null };
  setApiKeys: (keys: { openai?: string | null; anthropic?: string | null }) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  // Initialize user ID from localStorage or generate new
  userId: (() => {
    const stored = localStorage.getItem('userId');
    if (stored) return stored;
    const newId = crypto.randomUUID();
    localStorage.setItem('userId', newId);
    return newId;
  })(),
  setUserId: (id) => {
    localStorage.setItem('userId', id);
    set({ userId: id });
  },

  threads: [],
  setThreads: (threads) => set({ threads }),
  addThread: (thread) => set((state) => ({ threads: [thread, ...state.threads] })),
  updateThread: (threadId, updates) =>
    set((state) => ({
      threads: state.threads.map((t) => (t.id === threadId ? { ...t, ...updates } : t)),
    })),
  removeThread: (threadId) =>
    set((state) => ({
      threads: state.threads.filter((t) => t.id !== threadId),
    })),

  currentThreadId: (() => {
    const stored = localStorage.getItem('currentThreadId');
    return stored || null;
  })(),
  setCurrentThreadId: (id) => {
    if (id) {
      localStorage.setItem('currentThreadId', id);
    } else {
      localStorage.removeItem('currentThreadId');
    }
    set({ currentThreadId: id, messages: [] });
  },

  messages: [],
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  updateMessage: (messageId, updates) =>
    set((state) => ({
      messages: state.messages.map((m) => (m.id === messageId ? { ...m, ...updates } : m)),
    })),

  streamingDraft: null,
  setStreamingDraft: (threadId, text) => set({ streamingDraft: { threadId, text } }),
  clearStreamingDraft: () => set({ streamingDraft: null }),

  thinkingBlock: null,
  setThinkingBlock: (threadId, content) => set({ thinkingBlock: { threadId, content } }),
  clearThinkingBlock: () => set({ thinkingBlock: null }),

  toolDrafts: [],
  addToolDraft: (threadId, name, input) =>
    set((state) => ({ toolDrafts: [...state.toolDrafts, { threadId, name, input }] })),
  removeToolDraft: (threadId, name) =>
    set((state) => ({ toolDrafts: state.toolDrafts.filter((t) => !(t.threadId === threadId && t.name === name)) })),
  clearToolDrafts: (threadId) =>
    set((state) => ({ toolDrafts: state.toolDrafts.filter((t) => t.threadId !== threadId) })),
  
  subagentDrafts: [],
  addSubagentDraft: (threadId, agent, content) =>
    set((state) => {
      const existing = state.subagentDrafts.find((s) => s.threadId === threadId && s.agent === agent);
      if (existing) {
        return {
          subagentDrafts: state.subagentDrafts.map((s) =>
            s.threadId === threadId && s.agent === agent ? { ...s, content } : s
          ),
        };
      }
      return { subagentDrafts: [...state.subagentDrafts, { threadId, agent, content }] };
    }),
  updateSubagentDraft: (threadId, agent, content) =>
    set((state) => ({
      subagentDrafts: state.subagentDrafts.map((s) =>
        s.threadId === threadId && s.agent === agent ? { ...s, content } : s
      ),
    })),
  clearSubagentDrafts: (threadId) =>
    set((state) => ({ subagentDrafts: state.subagentDrafts.filter((s) => s.threadId !== threadId) })),
  finalizeSubagentDraft: (threadId, agent) =>
    set((state) => {
      const draft = state.subagentDrafts.find((s) => s.threadId === threadId && s.agent === agent);
      if (draft && draft.content.trim()) {
        // Move draft to segment and clear draft
        const now = Date.now();
        const segmentId = `segment-${now}-${Math.random().toString(36).substr(2, 9)}`;
        return {
          subagentSegments: [...state.subagentSegments, { threadId, agent, content: draft.content, id: segmentId, timestamp: now }],
          subagentDrafts: state.subagentDrafts.filter((s) => !(s.threadId === threadId && s.agent === agent)),
        };
      }
      // Just clear the draft if no content
      return {
        subagentDrafts: state.subagentDrafts.filter((s) => !(s.threadId === threadId && s.agent === agent)),
      };
    }),
  
  subagentSegments: [],
  clearSubagentSegments: (threadId) =>
    set((state) => ({ subagentSegments: state.subagentSegments.filter((s) => s.threadId !== threadId) })),
  removeSubagentSegments: (threadId, agents) =>
    set((state) => {
      const agentSet = new Set(agents);
      return {
        subagentSegments: state.subagentSegments.filter(
          (s) => !(s.threadId === threadId && agentSet.has(s.agent))
        ),
      };
    }),
  
  artifactBubbles: [],
  addArtifactBubble: (threadId, toolName, artifacts) =>
    set((state) => ({ artifactBubbles: [...state.artifactBubbles, { threadId, toolName, artifacts }] })),
  clearArtifactBubbles: (threadId) =>
    set((state) => ({ artifactBubbles: state.artifactBubbles.filter((a) => a.threadId !== threadId) })),

  isSidebarOpen: true,
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),

  theme: (localStorage.getItem('theme') as any) || 'auto',
  setTheme: (theme) => {
    localStorage.setItem('theme', theme);
    set({ theme });
  },

  sidebarWidth: parseInt(localStorage.getItem('sidebarWidth') || '256', 10),
  setSidebarWidth: (px) => {
    const clamped = Math.max(220, Math.min(800, px));
    localStorage.setItem('sidebarWidth', String(clamped));
    set({ sidebarWidth: clamped });
  },

  isArtifactsPanelOpen: false,
  toggleArtifactsPanel: () => set((state) => ({ isArtifactsPanelOpen: !state.isArtifactsPanelOpen })),
  artifactsPanelWidth: parseInt(localStorage.getItem('artifactsPanelWidth') || '400', 10),
  setArtifactsPanelWidth: (px) => {
    const clamped = Math.max(300, Math.min(800, px));
    localStorage.setItem('artifactsPanelWidth', String(clamped));
    set({ artifactsPanelWidth: clamped });
  },
  reports: [],
  setReports: (reports) => set({ reports }),
  currentReport: null,
  currentReportTitle: null,
  setCurrentReport: (report, title) =>
    set((state) => {
      if (report && title) {
        const filtered = state.reports.filter((r) => r.title !== title);
        return {
          currentReport: report,
          currentReportTitle: title ?? null,
          reports: [{ title, content: report }, ...filtered],
        };
      }
      return {
        currentReport: report,
        currentReportTitle: title ?? null,
        reports: [],
      };
    }),
  todos: [],
  setTodos: (todos) => set({ todos }),
  
  analysisScore: null,
  analysisStatus: null,
  setAnalysisScore: (score) => set({ analysisScore: score }),
  setAnalysisStatus: (status) => set({ analysisStatus: status }),

  defaultConfig: JSON.parse(localStorage.getItem('defaultConfig') || '{"model":null,"temperature":null,"system_prompt":null,"context_window":null}'),
  setDefaultConfig: (updates) => {
    set((state) => {
      const newConfig = { ...state.defaultConfig, ...updates };
      localStorage.setItem('defaultConfig', JSON.stringify(newConfig));
      return { defaultConfig: newConfig };
    });
  },

  contextUsage: { tokensUsed: 0, maxTokens: 30000 },
  setContextUsage: (tokensUsed, maxTokens) => set({ contextUsage: { tokensUsed, maxTokens } }),
  isSummarizing: false,
  setIsSummarizing: (value) => set({ isSummarizing: value }),
  isReviewing: false,
  setIsReviewing: (value) => set({ isReviewing: value }),

  toasts: [],
  addToast: (type, title, message, duration) => set((state) => ({
    toasts: [...state.toasts, { id: Date.now().toString(), type, title, message, duration }]
  })),
  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter(toast => toast.id !== id)
  })),

  selectedThreadIds: new Set(),
  toggleThreadSelection: (threadId) =>
    set((state) => {
      const newSelected = new Set(state.selectedThreadIds);
      if (newSelected.has(threadId)) {
        newSelected.delete(threadId);
      } else {
        newSelected.add(threadId);
      }
      return { selectedThreadIds: newSelected };
    }),
  selectAllThreads: (threadIds) => set({ selectedThreadIds: new Set(threadIds) }),
  clearThreadSelection: () => set({ selectedThreadIds: new Set() }),

  apiKeys: { openai: null, anthropic: null },
  setApiKeys: (updates) => {
    set((state) => ({
      apiKeys: { ...state.apiKeys, ...updates }
    }));
  },
}));

