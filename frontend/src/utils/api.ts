/**
 * API client utilities for backend communication.
 * Uses VITE_API_URL environment variable in production, falls back to /api for local dev.
 */

import type { Thread, Message, ThreadConfig } from '@/types/api';

const BASE_URL = import.meta.env.VITE_API_URL || '/api';

// ===== Threads =====

export async function createThread(userId: string, title?: string): Promise<Thread> {
  const res = await fetch(`${BASE_URL}/threads`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, title }),
  });
  if (!res.ok) throw new Error(`Failed to create thread: ${res.statusText}`);
  return res.json();
}

export async function listThreads(userId: string, limit = 20, includeArchived = false): Promise<Thread[]> {
  const res = await fetch(`${BASE_URL}/threads?user_id=${userId}&limit=${limit}&include_archived=${includeArchived}`);
  if (!res.ok) throw new Error(`Failed to list threads: ${res.statusText}`);
  return res.json();
}

export async function getThread(threadId: string): Promise<Thread> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}`);
  if (!res.ok) throw new Error(`Failed to get thread: ${res.statusText}`);
  return res.json();
}

export async function updateThreadTitle(threadId: string, title: string): Promise<Thread> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/title`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error(`Failed to update title: ${res.statusText}`);
  return res.json();
}

export async function autoGenerateTitle(threadId: string): Promise<Thread> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/title/auto`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to auto-generate title: ${res.statusText}`);
  return res.json();
}

// ===== Messages =====

export async function listMessages(threadId: string, limit = 50): Promise<Message[]> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/messages?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to list messages: ${res.statusText}`);
  return res.json();
}

// ===== Thread lifecycle =====

export async function archiveThread(threadId: string): Promise<Thread> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/archive`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to archive thread: ${res.statusText}`);
  return res.json();
}

export async function unarchiveThread(threadId: string): Promise<Thread> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/unarchive`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to unarchive thread: ${res.statusText}`);
  return res.json();
}

export async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Failed to delete thread: ${res.statusText}`);
}

// ===== Config =====

export async function getDefaultConfig(): Promise<ThreadConfig> {
  const res = await fetch(`${BASE_URL}/config/defaults`);
  if (!res.ok) throw new Error(`Failed to get default config: ${res.statusText}`);
  return res.json();
}

export async function getThreadConfig(threadId: string): Promise<ThreadConfig> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/config`);
  if (!res.ok) throw new Error(`Failed to get config: ${res.statusText}`);
  return res.json();
}

export async function updateThreadConfig(threadId: string, config: Partial<ThreadConfig>): Promise<ThreadConfig> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Failed to update config: ${res.statusText}`);
  return res.json();
}

// ===== Thread State =====

export async function getThreadState(threadId: string): Promise<{ token_count: number; context_window: number; analysis_objectives: string[] }> {
  const res = await fetch(`${BASE_URL}/threads/${threadId}/state`);
  if (!res.ok) throw new Error(`Failed to get thread state: ${res.statusText}`);
  return res.json();
}

// ===== API Key Management =====

export interface APIKeys {
  openai_key?: string | null;
  anthropic_key?: string | null;
}

export async function getUserApiKeys(userId: string): Promise<APIKeys> {
  const res = await fetch(`${BASE_URL}/users/${userId}/api-keys`);
  if (!res.ok) throw new Error(`Failed to get API keys: ${res.statusText}`);
  return res.json();
}

export async function saveUserApiKeys(userId: string, keys: APIKeys): Promise<APIKeys> {
  const res = await fetch(`${BASE_URL}/users/${userId}/api-keys`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(keys),
  });
  if (!res.ok) throw new Error(`Failed to save API keys: ${res.statusText}`);
  return res.json();
}

