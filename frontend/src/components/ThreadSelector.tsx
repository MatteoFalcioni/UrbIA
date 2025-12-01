/**
 * ThreadSelector: Top bar for thread selection and management.
 * Shows current thread title, dropdown for switching, and controls.
 */

import { useState, useEffect, useRef } from 'react';
import { ChevronDown, Plus, Sun, Moon, MessageCircle, Pencil, Archive, ArchiveRestore, Trash2, CheckSquare, Square, Menu } from 'lucide-react';
import { SignInButton, UserButton, useUser } from '@clerk/clerk-react';
import { useChatStore } from '@/store/chatStore';
import { createThread, updateThreadTitle, archiveThread, unarchiveThread, deleteThread, listThreads, listMessages } from '@/utils/api';
import { AnimatedTitle } from './AnimatedTitle';

interface ThreadSelectorProps {
  onCollapse: () => void;
}

export function ThreadSelector({ onCollapse }: ThreadSelectorProps) {
  const { isSignedIn } = useUser();
  const userId = useChatStore((state) => state.userId);
  const threads = useChatStore((state) => state.threads);
  const addThread = useChatStore((state) => state.addThread);
  const updateThread = useChatStore((state) => state.updateThread);
  const removeThread = useChatStore((state) => state.removeThread);
  const setThreads = useChatStore((state) => state.setThreads);
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const setCurrentThreadId = useChatStore((state) => state.setCurrentThreadId);
  const theme = useChatStore((state) => state.theme);
  const setTheme = useChatStore((state) => state.setTheme);
  const setContextUsage = useChatStore((state) => state.setContextUsage);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  const setCurrentReport = useChatStore((state) => state.setCurrentReport);
  const setTodos = useChatStore((state) => state.setTodos);
  
  // Bulk selection
  const selectedThreadIds = useChatStore((state) => state.selectedThreadIds);
  const toggleThreadSelection = useChatStore((state) => state.toggleThreadSelection);
  const selectAllThreads = useChatStore((state) => state.selectAllThreads);
  const clearThreadSelection = useChatStore((state) => state.clearThreadSelection);

  const [isOpen, setIsOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const [bulkMode, setBulkMode] = useState(false);

  const currentThread = threads.find(t => t.id === currentThreadId);
  const deletingThread = threads.find(t => t.id === deletingThreadId);

  // Fetch threads on mount
  useEffect(() => {
    async function loadThreads() {
      try {
        const fetchedThreads = await listThreads(userId, 50, false);
        setThreads(fetchedThreads);
      } catch (err) {
        console.error('Failed to load threads:', err);
      }
    }
    loadThreads();
  }, [userId, setThreads]);

  // Auto-select most recent thread on mount (only if no thread is currently selected)
  useEffect(() => {
    // Only run if threads are loaded and no thread is selected
    // Don't auto-select if we already have a thread selected (preserves selection when navigating)
    if (threads.length === 0 || currentThreadId) return;
    
    // Select the most recent thread (first in list)
    const mostRecentThread = threads[0];
    if (mostRecentThread) {
      console.log('ThreadSelector: Auto-selecting most recent thread:', mostRecentThread.id);
      setCurrentThreadId(mostRecentThread.id);
      setContextUsage(0, defaultConfig.context_window ?? 30000);
    }
  }, [threads.length, currentThreadId, setCurrentThreadId, setContextUsage, defaultConfig.context_window]);

  // Auto-create thread if none exist (only once, after threads are loaded)
  const hasAttemptedAutoCreate = useRef(false);
  useEffect(() => {
    // Only run once after initial load if no threads exist
    if (threads.length > 0 || currentThreadId || isCreating || hasAttemptedAutoCreate.current) return;
    
    // Mark that we've attempted auto-create to prevent multiple calls
    hasAttemptedAutoCreate.current = true;
    
    async function autoCreateThread() {
      setIsCreating(true);
      try {
        const newThread = await createThread(userId, 'New chat');
        addThread(newThread);
        setCurrentThreadId(newThread.id);
        setContextUsage(0, defaultConfig.context_window ?? 30000);
      } catch (err) {
        console.error('Failed to auto-create thread:', err);
        // Reset flag on error so we can retry
        hasAttemptedAutoCreate.current = false;
      } finally {
        setIsCreating(false);
      }
    }
    
    autoCreateThread();
  }, [threads.length, currentThreadId, isCreating, userId, addThread, setCurrentThreadId, setContextUsage, defaultConfig.context_window]);

  /**
   * Toggle between light and dark themes only
   */
  function toggleTheme() {
    if (theme === 'light') setTheme('dark');
    else setTheme('light');
  }

  // Create new thread handler
  async function handleCreateThread() {
    setIsCreating(true);
    try {
      // Delete current thread if it has no messages
      if (currentThreadId) {
        try {
          const messages = await listMessages(currentThreadId);
          if (messages.length === 0) {
            await deleteThread(currentThreadId);
            removeThread(currentThreadId);
          }
        } catch (err) {
          console.error('Failed to check/delete empty thread:', err);
          // Continue creating new thread even if deletion fails
        }
      }

      const newThread = await createThread(userId, 'New chat');
      addThread(newThread);
      setCurrentThreadId(newThread.id);
      // Reset context usage for new thread
      setContextUsage(0, defaultConfig.context_window ?? 30000);
      setIsOpen(false);
    } catch (err) {
      alert('Failed to create thread');
    } finally {
      setIsCreating(false);
    }
  }

  // Select thread handler
  async function handleSelectThread(threadId: string) {
    // Delete current thread if it has no messages before switching
    if (currentThreadId && currentThreadId !== threadId) {
      try {
        const messages = await listMessages(currentThreadId);
        if (messages.length === 0) {
          await deleteThread(currentThreadId);
          removeThread(currentThreadId);
        }
      } catch (err) {
        console.error('Failed to check/delete empty thread:', err);
        // Continue switching threads even if deletion fails
      }
    }

    setCurrentThreadId(threadId);
    setIsOpen(false);
    // Fetch actual context usage, objectives, and report from LangGraph state
    try {
      const { getThreadState } = await import('@/utils/api');
      const state = await getThreadState(threadId);
      setContextUsage(state.token_count, state.context_window);
      setTodos(state.todos || []);
      
      // Load report if available
      if (state.report_content && state.report_title) {
        setCurrentReport(state.report_content, state.report_title);
      } else {
        setCurrentReport(null, null);
      }
    } catch (err: any) {
      // Ignore 404 errors when fetching thread state (new thread)
      if (err?.response?.status !== 404) {
        console.error('Failed to fetch thread state:', err);
      }
      setContextUsage(0, defaultConfig.context_window ?? 30000);
      setTodos([]);
      setCurrentReport(null, null);
    }
  }

  // Rename thread handler
  function startEditingThread(threadId: string, currentTitle: string, e: React.MouseEvent) {
    e.stopPropagation();
    setEditingThreadId(threadId);
    setEditingTitle(currentTitle || '');
  }

  async function handleRenameThread(threadId: string) {
    if (!editingTitle.trim()) return;
    try {
      const updated = await updateThreadTitle(threadId, editingTitle.trim());
      updateThread(threadId, updated);
      setEditingThreadId(null);
    } catch (err) {
      alert('Failed to rename thread');
    }
  }

  // Archive/Unarchive thread handler
  async function handleToggleArchive(threadId: string, isArchived: boolean, e: React.MouseEvent) {
    e.stopPropagation();
    try {
      const updated = isArchived ? await unarchiveThread(threadId) : await archiveThread(threadId);
      updateThread(threadId, updated);
    } catch (err) {
      alert('Failed to archive/unarchive thread');
    }
  }

  // Delete thread handler - shows confirmation modal
  function showDeleteConfirmation(threadId: string, e: React.MouseEvent) {
    e.stopPropagation();
    setDeletingThreadId(threadId);
  }

  async function confirmDeleteThread() {
    if (!deletingThreadId) return;
    try {
      await deleteThread(deletingThreadId);
      setThreads(threads.filter(t => t.id !== deletingThreadId));
      if (currentThreadId === deletingThreadId) {
        setCurrentThreadId(null);
      }
      setDeletingThreadId(null);
    } catch (err) {
      alert('Failed to delete thread');
      setDeletingThreadId(null);
    }
  }

  // Bulk delete handler
  async function handleBulkDelete() {
    if (selectedThreadIds.size === 0) return;
    if (!confirm(`Delete ${selectedThreadIds.size} thread(s)? This cannot be undone.`)) return;

    try {
      await Promise.all(Array.from(selectedThreadIds).map((id) => deleteThread(id)));
      setThreads(threads.filter((t) => !selectedThreadIds.has(t.id)));
      if (currentThreadId && selectedThreadIds.has(currentThreadId)) {
        setCurrentThreadId(null);
      }
      clearThreadSelection();
      setBulkMode(false);
    } catch (err) {
      alert('Failed to delete some threads');
    }
  }

  // Bulk archive handler
  async function handleBulkArchive() {
    if (selectedThreadIds.size === 0) return;

    try {
      await Promise.all(Array.from(selectedThreadIds).map((id) => archiveThread(id)));
      // Update threads to show as archived
      setThreads(threads.map((t) => 
        selectedThreadIds.has(t.id) ? { ...t, archived_at: new Date().toISOString() } : t
      ));
      clearThreadSelection();
      setBulkMode(false);
    } catch (err) {
      alert('Failed to archive some threads');
    }
  }

  // Toggle select all
  function handleSelectAll() {
    if (selectedThreadIds.size === threads.length) {
      clearThreadSelection();
    } else {
      selectAllThreads(threads.map((t) => t.id));
    }
  }

  return (
    <div className="relative">
      {/* Top bar */}
      <div className="flex items-center justify-between gap-2 p-3 border-b border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800">
        {/* Left side: Collapse button + Thread selector */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {/* Collapse button */}
          <button
            onClick={onCollapse}
            className="p-2 rounded-xl border border-gray-200 dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 shadow-sm hover:shadow-md group flex-shrink-0"
            title="Hide sidebar"
          >
            <Menu size={16} className="text-gray-500 dark:text-slate-400 group-hover:text-gray-700 dark:group-hover:text-slate-200 transition-colors" />
          </button>
          
          {/* Thread selector */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-gray-200 dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 min-w-0 shadow-sm hover:shadow-md flex-1"
          >
            <MessageCircle size={16} className="text-gray-500 dark:text-slate-400 flex-shrink-0" />
            <AnimatedTitle 
              title={currentThread?.title || 'Select a thread'} 
              className="truncate text-sm font-medium text-gray-700 dark:text-slate-200"
              duration={300}
            />
            <ChevronDown size={14} className="text-gray-400 dark:text-slate-500 flex-shrink-0" />
          </button>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={handleCreateThread}
            disabled={isCreating}
            className="p-2 rounded-xl border border-gray-200 dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 disabled:opacity-50 shadow-sm hover:shadow-md"
            title="New Thread"
          >
            <Plus size={16} className="text-gray-500 dark:text-slate-400" />
          </button>
          
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl border border-gray-200 dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 shadow-sm hover:shadow-md"
            title={`Theme: ${theme}`}
          >
            {theme === 'dark' ? (
              <Moon size={16} className="text-gray-500 dark:text-slate-400" />
            ) : (
              <Sun size={16} className="text-gray-500 dark:text-slate-400" />
            )}
          </button>

          {/* Authentication */}
          {isSignedIn ? (
            <UserButton 
              appearance={{
                elements: {
                  avatarBox: "w-8 h-8"
                }
              }}
            />
          ) : (
            <SignInButton mode="modal">
              <button
                className="px-3 py-1.5 rounded-xl border border-gray-200 dark:border-slate-700 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 shadow-sm hover:shadow-md text-sm font-medium text-gray-700 dark:text-slate-200"
                title="Sign in"
              >
                Sign in
              </button>
            </SignInButton>
          )}
        </div>
      </div>

      {/* Dropdown panel */}
      {isOpen && (
        <div className="absolute top-full left-0 z-50 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl shadow-lg max-h-80 overflow-y-auto min-w-80 max-w-96">
          {threads.length === 0 ? (
            <div className="p-4 text-center text-gray-500 dark:text-slate-400 text-sm">
              No threads yet. Create one to start chatting!
            </div>
          ) : (
            <>
              {/* Bulk action header */}
              <div className="px-4 py-2 border-b border-gray-200 dark:border-slate-700 flex items-center justify-between">
                {bulkMode ? (
                  <>
                    <button
                      onClick={handleSelectAll}
                      className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                    >
                      {selectedThreadIds.size === threads.length ? (
                        <CheckSquare size={16} className="text-blue-600" />
                      ) : (
                        <Square size={16} />
                      )}
                      <span>Select All ({selectedThreadIds.size}/{threads.length})</span>
                    </button>
                    <div className="flex items-center gap-2">
                      {selectedThreadIds.size > 0 && (
                        <>
                          <button
                            onClick={handleBulkArchive}
                            className="px-2 py-1 text-xs bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded flex items-center gap-1"
                            title="Archive selected"
                          >
                            <Archive size={12} />
                            Archive
                          </button>
                          <button
                            onClick={handleBulkDelete}
                            className="px-2 py-1 text-xs bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30 text-red-600 dark:text-red-400 rounded flex items-center gap-1"
                            title="Delete selected"
                          >
                            <Trash2 size={12} />
                            Delete
                          </button>
                        </>
                      )}
                      <button
                        onClick={() => {
                          setBulkMode(false);
                          clearThreadSelection();
                        }}
                        className="px-2 py-1 text-xs hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                ) : (
                  <button
                    onClick={() => setBulkMode(true)}
                    className="text-sm text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200 hover:underline transition-colors"
                  >
                    Select Multiple
                  </button>
                )}
              </div>
              
              <div className="py-2">
                {threads.map((thread) => (
                <div
                  key={thread.id}
                  className={`group relative px-4 py-3 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 ${
                    thread.id === currentThreadId ? 'bg-slate-100 dark:bg-slate-700 border-l-4 border-slate-600 dark:border-slate-400' : ''
                  }`}
                >
                  {editingThreadId === thread.id ? (
                    // Edit mode
                    <div className="flex items-center gap-2">
                      <MessageCircle size={16} className="text-gray-400 dark:text-slate-500 flex-shrink-0" />
                      <input
                        type="text"
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleRenameThread(thread.id);
                          } else if (e.key === 'Escape') {
                            setEditingThreadId(null);
                          }
                        }}
                        onBlur={() => handleRenameThread(thread.id)}
                        autoFocus
                        className="flex-1 px-2 py-1 text-sm border border-blue-500 rounded bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 focus:outline-none"
                      />
                    </div>
                  ) : (
                    // Normal mode
                  <div className="flex items-center gap-2">
                    {bulkMode && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleThreadSelection(thread.id);
                        }}
                        className="flex-shrink-0"
                      >
                        {selectedThreadIds.has(thread.id) ? (
                          <CheckSquare size={16} className="text-blue-600" />
                        ) : (
                          <Square size={16} className="text-gray-400 dark:text-slate-500" />
                        )}
                      </button>
                    )}
                    <button
                      onClick={() => bulkMode ? toggleThreadSelection(thread.id) : handleSelectThread(thread.id)}
                      className="flex-1 flex items-center gap-3 min-w-0"
                    >
                      <MessageCircle size={16} className="text-gray-400 dark:text-slate-500 flex-shrink-0" />
                      <span className="truncate text-sm font-medium text-gray-700 dark:text-slate-200">
                        {thread.title || 'Untitled'}
                      </span>
                      {thread.archived_at && (
                        <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-slate-700 rounded-full flex-shrink-0 text-gray-600 dark:text-slate-300">
                          Archived
                        </span>
                      )}
                    </button>
                      
                      {/* Action buttons - show on hover (hide in bulk mode) */}
                      {!bulkMode && (
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                          onClick={(e) => startEditingThread(thread.id, thread.title || '', e)}
                          className="p-1 hover:bg-gray-100 dark:hover:bg-slate-600 rounded transition-colors"
                          title="Rename"
                        >
                          <Pencil size={14} className="text-gray-400 dark:text-slate-500" />
                        </button>
                        <button
                          onClick={(e) => handleToggleArchive(thread.id, !!thread.archived_at, e)}
                          className="p-1 hover:bg-gray-100 dark:hover:bg-slate-600 rounded transition-colors"
                          title={thread.archived_at ? 'Unarchive' : 'Archive'}
                        >
                          {thread.archived_at ? (
                            <ArchiveRestore size={14} className="text-gray-400 dark:text-slate-500" />
                          ) : (
                            <Archive size={14} className="text-gray-400 dark:text-slate-500" />
                          )}
                        </button>
                        <button
                          onClick={(e) => showDeleteConfirmation(thread.id, e)}
                          className="p-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                          title="Delete"
                        >
                          <Trash2 size={14} className="text-red-400 dark:text-red-400" />
                        </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              </div>
            </>
          )}
          
          {/* Create new thread button */}
          <div className="border-t border-gray-200 dark:border-slate-700 p-3">
            <button
              onClick={handleCreateThread}
              disabled={isCreating}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-900 disabled:bg-gray-400 text-white rounded-xl transition-all duration-200 text-sm shadow-sm hover:shadow-md"
            >
              <Plus size={16} />
              <span>{isCreating ? 'Creating...' : 'New Thread'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deletingThreadId && (
        <>
          <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center" onClick={() => setDeletingThreadId(null)}>
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl max-w-md w-full mx-4 p-6" onClick={(e) => e.stopPropagation()}>
              {/* Header */}
              <div className="flex items-start gap-4 mb-4">
                <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-full">
                  <Trash2 size={24} className="text-red-600 dark:text-red-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-100 mb-1">
                    Delete Thread
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-slate-400">
                    Are you sure you want to delete "{deletingThread?.title || 'Untitled'}"?
                  </p>
                </div>
              </div>

              {/* Warning message */}
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 mb-6">
                <p className="text-sm text-red-800 dark:text-red-300">
                  This action cannot be undone. All messages and data in this thread will be permanently deleted.
                </p>
              </div>

              {/* Actions */}
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setDeletingThreadId(null)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDeleteThread}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors shadow-sm"
                >
                  Delete Thread
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}


