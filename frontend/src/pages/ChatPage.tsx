/**
 * ChatPage: Main chat interface page
 * Layout: [Sidebar] [MessageView]
 */

import { useRef, useState } from 'react';
import { Settings, MessageSquare, PanelRightOpen } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ArtifactDisplay } from '@/components/ArtifactDisplay';
import { TodoListDropdown } from '@/components/TodoListDropdown';
import { ArtifactsPanel } from '@/components/ArtifactsPanel';
import { ApiKeyWarning } from '@/components/ApiKeyWarning';
import { ScoreBar } from '@/components/ScoreBar';

export function ChatPage() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const sidebarWidth = useChatStore((s) => s.sidebarWidth);
  const setSidebarWidth = useChatStore((s) => s.setSidebarWidth);
  const isArtifactsPanelOpen = useChatStore((s) => s.isArtifactsPanelOpen);
  const toggleArtifactsPanel = useChatStore((s) => s.toggleArtifactsPanel);
  const artifactsPanelWidth = useChatStore((s) => s.artifactsPanelWidth);
  const setArtifactsPanelWidth = useChatStore((s) => s.setArtifactsPanelWidth);
  const analysisScore = useChatStore((s) => s.analysisScore);
  const analysisStatus = useChatStore((s) => s.analysisStatus);
  
  const isResizing = useRef(false);
  const startWidth = useRef(0);
  const startX = useRef(0);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  
  const isResizingRight = useRef(false);
  const startWidthRight = useRef(0);
  const startXRight = useRef(0);

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

  // Handle resize for right panel (artifacts)
  function handleResizeStartRight(e: React.MouseEvent) {
    e.preventDefault();
    isResizingRight.current = true;
    startXRight.current = e.clientX;
    startWidthRight.current = artifactsPanelWidth;

    function handleMouseMoveRight(e: MouseEvent) {
      if (!isResizingRight.current) return;
      const delta = startXRight.current - e.clientX; // Reversed for right panel
      const newWidth = startWidthRight.current + delta;
      setArtifactsPanelWidth(newWidth);
    }

    function handleMouseUpRight() {
      isResizingRight.current = false;
      document.removeEventListener('mousemove', handleMouseMoveRight);
      document.removeEventListener('mouseup', handleMouseUpRight);
    }

    document.addEventListener('mousemove', handleMouseMoveRight);
    document.addEventListener('mouseup', handleMouseUpRight);
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-slate-900 text-gray-700 dark:text-slate-200 overflow-hidden">
      {/* API Key Warning Modal */}
      <ApiKeyWarning />
      
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
        {/* Header with score bar, todos dropdown and buttons */}
        <div className="flex items-center justify-between gap-2 p-2 border-b border-gray-200 dark:border-slate-700 flex-shrink-0">
          {/* Left side: Score bar */}
          <div className="flex-1 min-w-0">
            <ScoreBar score={analysisScore} status={analysisStatus} />
          </div>
          
          {/* Right side: Controls */}
          <div className="flex items-center gap-2">
            <TodoListDropdown />
            <button
            onClick={toggleArtifactsPanel}
            className="p-2 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-sm hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 flex items-center justify-center"
            title="Toggle artifacts panel"
          >
            <PanelRightOpen size={18} className="text-gray-500 dark:text-slate-400" />
          </button>
          <button
            onClick={() => window.open('/settings', '_blank')}
            className="p-2 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg shadow-sm hover:bg-gray-50 dark:hover:bg-slate-700 transition-all duration-200 flex items-center justify-center"
            title={currentThreadId ? 'Thread Settings (opens in new tab)' : 'Default Settings (opens in new tab)'}
          >
            <Settings size={18} className="text-gray-500 dark:text-slate-400" />
          </button>
          </div>
        </div>
        
        {/* Artifact display area */}
        <div className="flex-1 overflow-hidden">
          <ArtifactDisplay />
        </div>
      </main>

      {/* Right sidebar: Artifacts panel (resizable) */}
      {isArtifactsPanelOpen && (
        <aside
          className="bg-white dark:bg-slate-800 border-l border-gray-200 dark:border-slate-700 relative flex flex-col"
          style={{ width: artifactsPanelWidth }}
        >
          {/* Resize handle (left side) */}
          <div
            onMouseDown={handleResizeStartRight}
            className="absolute top-0 left-0 w-2 h-full cursor-col-resize hover:bg-amber-400/20 bg-transparent z-10"
          />
          <ArtifactsPanel />
        </aside>
      )}
    </div>
  );
}

