/**
 * ModelSelector: Dropdown component for selecting the model.
 * Opens upward and shows the current model.
 */

import { useState, useRef, useEffect } from 'react';
import { ChevronUp } from 'lucide-react';
import { getThreadConfig, updateThreadConfig } from '@/utils/api';
import { useChatStore } from '@/store/chatStore';

const MODELS = [
  { value: 'gpt-4.1', label: 'GPT-4.1' },
  { value: 'claude-sonnet-4-5', label: 'Claude Sonnet 4.5' },
  { value: 'claude-haiku-4-5', label: 'Claude Haiku 4.5' },
] as const;

export function ModelSelector() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const defaultConfigModel = useChatStore((state) => state.defaultConfig.model);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  
  const [isOpen, setIsOpen] = useState(false);
  const [currentModel, setCurrentModel] = useState<string>('gpt-4.1');
  const [isLoading, setIsLoading] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load current model
  const loadModel = async () => {
    if (!currentThreadId) {
      // No thread: use default config
      const model = defaultConfigModel || 'gpt-4.1';
      console.log('ModelSelector: No thread, using defaultConfigModel:', model);
      setCurrentModel(model);
      return;
    }

    try {
      const config = await getThreadConfig(currentThreadId);
      const model = config.model || defaultConfigModel || 'gpt-4.1';
      console.log('ModelSelector: Loaded thread config, model:', model);
      setCurrentModel(model);
    } catch (err) {
      console.error('Failed to load thread config:', err);
      const model = defaultConfigModel || 'gpt-4.1';
      setCurrentModel(model);
    }
  };

  // Load current model when thread changes
  useEffect(() => {
    if (!currentThreadId) {
      // No thread: directly use defaultConfigModel (no API call needed)
      const model = defaultConfigModel || 'gpt-4.1';
      console.log('ModelSelector: No thread, using defaultConfigModel:', model);
      setCurrentModel(model);
    } else {
      // Thread selected: load from API
      loadModel();
    }
  }, [currentThreadId, defaultConfigModel]);

  // Reload model when opening dropdown to ensure it's up-to-date (only if thread exists)
  useEffect(() => {
    if (isOpen && currentThreadId) {
      loadModel();
    } else if (isOpen && !currentThreadId) {
      // No thread: just update from defaultConfigModel
      const model = defaultConfigModel || 'gpt-4.1';
      setCurrentModel(model);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // Reload model when page becomes visible (e.g., returning from settings) - only if thread exists
  useEffect(() => {
    if (!currentThreadId) return; // Skip if no thread
    
    function handleVisibilityChange() {
      if (document.visibilityState === 'visible') {
        loadModel();
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentThreadId]);

  // Reload model when window gains focus (e.g., switching tabs back) - only if thread exists
  useEffect(() => {
    if (!currentThreadId) return; // Skip if no thread
    
    function handleFocus() {
      loadModel();
    }

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentThreadId]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  async function handleModelChange(model: string) {
    if (model === currentModel) {
      setIsOpen(false);
      return;
    }

    setIsLoading(true);
    try {
      if (currentThreadId) {
        // Update thread config
        await updateThreadConfig(currentThreadId, { model });
        setCurrentModel(model);
      } else {
        // No thread: update default config in store
        useChatStore.getState().setDefaultConfig({
          ...defaultConfig,
          model,
        });
        setCurrentModel(model);
      }
    } catch (err) {
      console.error('Failed to update model:', err);
      alert('Failed to update model');
    } finally {
      setIsLoading(false);
      setIsOpen(false);
    }
  }

  const currentModelLabel = MODELS.find(m => m.value === currentModel)?.label || currentModel;

  return (
    <div ref={dropdownRef} className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
        className="px-2 py-1 text-xs rounded-md transition-all duration-200 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
        style={{
          border: '1px solid var(--border)',
          backgroundColor: 'var(--bg-secondary)',
          color: 'var(--text-primary)',
          opacity: 0.6,
        }}
      >
        <span>{currentModelLabel}</span>
        <ChevronUp 
          size={12} 
          className={`transition-transform duration-200 ${isOpen ? '' : 'rotate-180'}`}
        />
      </button>

      {isOpen && (
        <div
          className="absolute bottom-full left-0 mb-2 min-w-[160px] rounded-md shadow-lg border z-50 overflow-hidden"
          style={{
            borderColor: 'var(--border)',
            backgroundColor: 'var(--bg-primary)',
          }}
        >
          {MODELS.map((model) => (
            <button
              key={model.value}
              type="button"
              onClick={() => handleModelChange(model.value)}
              className="w-full px-3 py-1.5 text-left text-xs transition-colors hover:bg-opacity-50 flex items-center justify-between"
              style={{
                backgroundColor: model.value === currentModel ? 'var(--bg-secondary)' : 'transparent',
                color: 'var(--text-primary)',
              }}
            >
              <span>{model.label}</span>
              {model.value === currentModel && (
                <span className="text-xs">✓</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

