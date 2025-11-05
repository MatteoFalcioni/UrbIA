/**
 * InterruptModal: Display and handle graph interrupts (HITL)
 * Shows different UI based on interrupt type (report approval, report review/edit)
 */

import { useState } from 'react';

interface InterruptModalProps {
  interruptData: any;
  onResume: (resumeValue: Record<string, any>) => void;
  onCancel: () => void;
}

export function InterruptModal({ interruptData, onResume, onCancel }: InterruptModalProps) {
  const [editInstructions, setEditInstructions] = useState('');
  
  // Detect which interrupt this is based on the data structure
  const isWriteReportApproval = interruptData && 
    typeof interruptData === 'string' && 
    interruptData.includes('write a report');
  
  const isReportReview = interruptData?.report && interruptData?.question;
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        
        {/* First Interrupt: Report Writing Approval */}
        {isWriteReportApproval && (
          <>
            <div className="p-6 border-b border-gray-200 dark:border-slate-700">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                📝 Report Writing Approval
              </h2>
            </div>
            <div className="p-6 flex-1 overflow-y-auto">
              <p className="text-gray-700 dark:text-slate-300 mb-4">
                The assistant has completed the analysis and wants to write a report. Do you approve?
              </p>
              <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4 text-sm text-amber-800 dark:text-amber-200">
                <strong>Note:</strong> Once approved, the assistant will generate a comprehensive report based on the analysis performed.
              </div>
            </div>
            <div className="p-6 border-t border-gray-200 dark:border-slate-700 flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-4 py-2 bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-300 dark:hover:bg-slate-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => onResume({ type: 'reject' })}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              >
                Reject
              </button>
              <button
                onClick={() => onResume({ type: 'accept' })}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                Approve
              </button>
            </div>
          </>
        )}
        
        {/* Second Interrupt: Report Review/Edit */}
        {isReportReview && (
          <>
            <div className="p-6 border-b border-gray-200 dark:border-slate-700">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                📄 Report Generated
              </h2>
              <p className="text-sm text-gray-500 dark:text-slate-400 mt-1">
                Review the report and approve or request changes
              </p>
            </div>
            <div className="p-6 flex-1 overflow-y-auto">
              {/* Report preview */}
              <div className="bg-gray-50 dark:bg-slate-900 rounded-lg p-4 mb-4 max-h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap text-sm text-gray-900 dark:text-slate-100 font-mono">
                  {interruptData.report}
                </pre>
              </div>
              
              {/* Edit instructions input */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                  Request Changes (optional):
                </label>
                <textarea
                  value={editInstructions}
                  onChange={(e) => setEditInstructions(e.target.value)}
                  placeholder="e.g., Add more details to the conclusion, translate the first section to Italian, make it more concise..."
                  className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={3}
                />
              </div>
            </div>
            <div className="p-6 border-t border-gray-200 dark:border-slate-700 flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-4 py-2 bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-300 dark:hover:bg-slate-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (editInstructions.trim()) {
                    onResume({ type: 'edit', edit_instructions: editInstructions });
                  } else {
                    alert('Please enter edit instructions or click Approve');
                  }
                }}
                disabled={!editInstructions.trim()}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Request Changes
              </button>
              <button
                onClick={() => onResume({ type: 'accept' })}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                Approve Report
              </button>
            </div>
          </>
        )}
        
        {/* Fallback for unknown interrupt type */}
        {!isWriteReportApproval && !isReportReview && (
          <>
            <div className="p-6 border-b border-gray-200 dark:border-slate-700">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                ⏸️ Approval Required
              </h2>
            </div>
            <div className="p-6 flex-1 overflow-y-auto">
              <p className="text-sm text-gray-600 dark:text-slate-400 mb-3">
                The assistant is waiting for approval:
              </p>
              <pre className="text-sm text-gray-700 dark:text-slate-300 whitespace-pre-wrap bg-gray-50 dark:bg-slate-900 p-4 rounded-lg overflow-x-auto">
                {JSON.stringify(interruptData, null, 2)}
              </pre>
            </div>
            <div className="p-6 border-t border-gray-200 dark:border-slate-700 flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-4 py-2 bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-300 dark:hover:bg-slate-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => onResume({ type: 'accept' })}
                className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                Continue
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}


