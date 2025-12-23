/**
 * ArtifactsPanel: Right sidebar for displaying artifacts (reports, reviews, etc.)
 */

import { X, FileText, ChevronDown, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useChatStore } from '@/store/chatStore';
import { useMemo, useState } from 'react';

export function ArtifactsPanel() {
  const togglePanel = useChatStore((state) => state.toggleArtifactsPanel);
  const reports = useChatStore((state) => state.reports);
  const currentReportTitle = useChatStore((state) => state.currentReportTitle);
  const [collapsedMap, setCollapsedMap] = useState<Record<string, boolean>>({});

  const orderedReports = useMemo(() => {
    if (!reports || reports.length === 0) return [];
    if (!currentReportTitle) return reports;
    return [
      ...reports.filter((r) => r.title === currentReportTitle),
      ...reports.filter((r) => r.title !== currentReportTitle),
    ];
  }, [reports, currentReportTitle]);

  return (
    <>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-slate-700 flex-shrink-0">
        <div className="flex items-center gap-2">
          <FileText size={18} className="text-gray-600 dark:text-slate-400" />
          <h2 className="text-lg font-semibold">Artifacts</h2>
        </div>
        <button
          onClick={togglePanel}
          className="p-1.5 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
          title="Close artifacts panel"
        >
          <X size={18} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {orderedReports.length > 0 ? (
          <div className="flex flex-col gap-4">
            {orderedReports.map((report) => {
              const isCollapsed = collapsedMap[report.title] ?? false;
              return (
                <div
                  key={report.title}
                  className="max-w-3xl bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-xl shadow-sm"
                >
                  <button
                    onClick={() =>
                      setCollapsedMap((prev) => ({
                        ...prev,
                        [report.title]: !isCollapsed,
                      }))
                    }
                    className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-gray-50 dark:hover:bg-slate-700/60 transition-colors rounded-t-xl"
                  >
                    <div className="flex items-center gap-2">
                      {isCollapsed ? (
                        <ChevronRight size={16} className="text-gray-500 dark:text-slate-400" />
                      ) : (
                        <ChevronDown size={16} className="text-gray-500 dark:text-slate-400" />
                      )}
                      <span className="text-sm font-semibold text-gray-800 dark:text-slate-100">
                        {report.title || 'Report'}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500 dark:text-slate-400">
                      {isCollapsed ? 'Show content' : 'Hide content'}
                    </span>
                  </button>

                  {!isCollapsed && (
                    <div className="px-5 pb-5">
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {report.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 dark:text-slate-400">
            <FileText size={48} className="mb-4 opacity-30" />
            <p className="text-sm">No artifacts yet</p>
            <p className="text-xs mt-1">Reports and analysis will appear here</p>
          </div>
        )}
      </div>
    </>
  );
}

