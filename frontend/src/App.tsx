/**
 * Main App component.
 * Layout: [Sidebar] [MessageView] [ConfigPanel?]
 */

import { useEffect, useRef, useState } from 'react';
import { Settings, MessageSquare } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ArtifactDisplay } from '@/components/ArtifactDisplay';
import { ConfigPanel } from '@/components/ConfigPanel';
import { ToastManager } from '@/components/Toast';
import { useClerkSync } from '@/hooks/useClerkSync';

function App() {
  // Sync Clerk authentication with chat store
  useClerkSync();
  const theme = useChatStore((state) => state.theme);
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const toggleConfigPanel = useChatStore((state) => state.toggleConfigPanel);
  const sidebarWidth = useChatStore((s) => s.sidebarWidth);
  const setSidebarWidth = useChatStore((s) => s.setSidebarWidth);
  const toasts = useChatStore((state) => state.toasts);
  const removeToast = useChatStore((state) => state.removeToast);
  const isResizing = useRef(false);
  const startWidth = useRef(0);
  const startX = useRef(0);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Apply theme to document root
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else if (theme === 'light') {
      root.classList.remove('dark');
    } else {
      // Auto: match system preference
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (isDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
  }, [theme]);

  // Handle resize start and attach listeners directly
  function handleResizeStart(e: React.MouseEvent) {
    e.preventDefault();
    isResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = sidebarWidth;

    function handleMouseMove(e: MouseEvent) {
      if (!isResizing.current) return;
      const delta = e.clientX - startX.current;
      const newWidth = startWidth.current + delta;
      setSidebarWidth(newWidth);
    }

    function handleMouseUp() {
      isResizing.current = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-slate-900 text-gray-700 dark:text-slate-200 overflow-hidden">
      {/* Sidebar: Thread list (resizable/collapsible) */}
      {!isSidebarCollapsed && (
        <aside
          className="bg-white dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 relative"
          style={{ width: sidebarWidth }}
        >
          <ChatSidebar onCollapse={() => setIsSidebarCollapsed(true)} />
          {/* Resize handle */}
          <div
            onMouseDown={handleResizeStart}
            className="absolute top-0 right-0 w-2 h-full cursor-col-resize hover:bg-amber-400/20 bg-transparent"
          />
        </aside>
      )}

      {/* Modern floating show sidebar button (only when collapsed) */}
      {isSidebarCollapsed && (
        <button
          onClick={() => setIsSidebarCollapsed(false)}
          className="fixed top-4 left-4 z-50 p-2 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm border border-gray-200/50 dark:border-slate-700/50 rounded-full shadow-lg hover:shadow-xl hover:bg-white dark:hover:bg-slate-800 transition-all duration-300 ease-out flex items-center justify-center group hover:scale-105 active:scale-95"
          title="Show sidebar"
        >
          <MessageSquare size={16} className="text-gray-600 dark:text-slate-300 group-hover:text-gray-800 dark:group-hover:text-slate-100 transition-colors duration-200" />
        </button>
      )}

      {/* Main content: Artifact Display */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header with config toggle */}
        <div className="flex items-center justify-end p-2 border-b border-gray-200 dark:border-slate-700 flex-shrink-0">
          <button
            onClick={toggleConfigPanel}
            className="p-2 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-sm hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 flex items-center justify-center"
            title={currentThreadId ? 'Thread Settings' : 'Default Settings'}
          >
            <Settings size={18} className="text-gray-500 dark:text-slate-400" />
          </button>
        </div>
        
        {/* Artifact display area */}
        <div className="flex-1 overflow-hidden">
          <ArtifactDisplay />
        </div>
      </main>

      {/* Right panel: Config (toggleable) */}
      <ConfigPanel />

      {/* Toast notifications */}
      <ToastManager toasts={toasts} onRemove={removeToast} />
    </div>
  );
}

export default App;

