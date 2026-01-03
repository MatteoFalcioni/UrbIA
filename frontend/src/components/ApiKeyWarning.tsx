/**
 * ApiKeyWarning: Modal that warns users if they haven't set up API keys yet.
 * Reads from the store (populated by useApiKeysLoader) instead of making a separate API call.
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, Settings } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '@/store/chatStore';

export function ApiKeyWarning() {
  const [showWarning, setShowWarning] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const userId = useChatStore((state) => state.userId);
  const apiKeys = useChatStore((state) => state.apiKeys);
  const navigate = useNavigate();

  useEffect(() => {
    // Wait a bit for useApiKeysLoader to populate the store
    const timer = setTimeout(() => {
      if (!userId) {
        setIsChecking(false);
        return;
      }
      
      // Check if any keys exist in the store
      const hasKeys = apiKeys.openai || apiKeys.anthropic;
      setShowWarning(!hasKeys);
      setIsChecking(false);
    }, 500); // Small delay to let the loader hook finish

    return () => clearTimeout(timer);
  }, [userId, apiKeys]);

  // Don't show anything while checking or if no warning needed
  if (isChecking || !showWarning) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full p-6 relative">
        {/* Warning Icon */}
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-amber-100 dark:bg-amber-900/30 rounded-full">
            <AlertTriangle size={24} className="text-amber-600 dark:text-amber-400" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            API Keys Required
          </h2>
        </div>

        {/* Message */}
        <p className="text-gray-600 dark:text-slate-300 mb-6">
          To use this application, you need to provide at least one API key (OpenAI or Anthropic). 
          Your keys are encrypted and stored securely.
        </p>

        {/* Action Button */}
        <button
          onClick={() => navigate('/settings')}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gray-900 dark:bg-slate-700 text-white rounded-lg hover:bg-gray-800 dark:hover:bg-slate-600 transition-colors font-medium"
        >
          <Settings size={18} />
          Go to Settings
        </button>

        {/* Info text */}
        <p className="text-sm text-gray-500 dark:text-slate-400 mt-4 text-center">
          You'll need an OpenAI or Anthropic API key to continue.
        </p>
      </div>
    </div>
  );
}

