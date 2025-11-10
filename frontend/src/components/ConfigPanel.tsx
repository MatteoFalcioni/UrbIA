/**
 * ConfigPanel: right-side panel for thread-specific or default configuration.
 * - When thread is selected: shows and saves thread-specific config
 * - When no thread: shows and saves default config for new threads
 */

import { useEffect, useState } from 'react';
import { X, Save, Settings, Eye, EyeOff, Key, Target } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { getDefaultConfig, getThreadConfig, updateThreadConfig, getUserApiKeys, saveUserApiKeys, getThreadState } from '@/utils/api';
import type { ThreadConfig } from '@/types/api';

export function ConfigPanel() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const isOpen = useChatStore((state) => state.isConfigPanelOpen);
  const toggleConfigPanel = useChatStore((state) => state.toggleConfigPanel);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  const setDefaultConfig = useChatStore((state) => state.setDefaultConfig);
  const contextUsage = useChatStore((state) => state.contextUsage);
  const setContextUsage = useChatStore((state) => state.setContextUsage);
  const userId = useChatStore((state) => state.userId);
  const setApiKeys = useChatStore((state) => state.setApiKeys);

  const [config, setConfig] = useState<ThreadConfig>({
    model: null,
    temperature: null,
    system_prompt: null,
    context_window: null,
    settings: null,
  });
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [analysisObjectives, setAnalysisObjectives] = useState<string[]>([]);
  
  // API Keys state
  const [apiKeyInputs, setApiKeyInputs] = useState({
    openai: '',
    anthropic: '',
  });
  const [showKeys, setShowKeys] = useState({
    openai: false,
    anthropic: false,
  });
  const [isSavingKeys, setIsSavingKeys] = useState(false);
  const [keysSaveStatus, setKeysSaveStatus] = useState<'idle' | 'success' | 'error'>('idle');

  // Load config when thread changes
  useEffect(() => {
    async function loadConfig() {
      if (!currentThreadId) {
        // No thread: fetch backend defaults from /config/defaults
        setAnalysisObjectives([]);
        try {
          const cfg = await getDefaultConfig();
          setConfig(cfg);
          // Also update store's defaultConfig with backend values
          setDefaultConfig({
            model: cfg.model,
            temperature: cfg.temperature,
            system_prompt: cfg.system_prompt,
            context_window: cfg.context_window,
          });
        } catch (err) {
          console.error('Failed to load default config:', err);
          // Fallback to localStorage defaults if API fails
          setConfig({
            model: defaultConfig.model,
            temperature: defaultConfig.temperature,
            system_prompt: defaultConfig.system_prompt,
            context_window: defaultConfig.context_window,
            settings: null,
          });
        }
        return;
      }

      // Thread selected: load thread-specific config and analysis objectives
      try {
        const cfg = await getThreadConfig(currentThreadId!);
        setConfig(cfg);
        
        // Load analysis objectives from thread state
        const state = await getThreadState(currentThreadId!);
        setAnalysisObjectives(state.analysis_objectives || []);
      } catch (err) {
        console.error('Failed to load config:', err);
        setAnalysisObjectives([]);
      }
    }
    loadConfig();
  }, [currentThreadId]);

  // Load API keys on mount
  useEffect(() => {
    async function loadApiKeys() {
      try {
        const keys = await getUserApiKeys(userId);
        // Set masked keys in inputs for display
        setApiKeyInputs({
          openai: keys.openai_key || '',
          anthropic: keys.anthropic_key || '',
        });
        // Update store with masked keys
        setApiKeys({
          openai: keys.openai_key || null,
          anthropic: keys.anthropic_key || null,
        });
      } catch (err) {
        console.error('Failed to load API keys:', err);
      }
    }
    loadApiKeys();
  }, [userId, setApiKeys]);

  // Refresh analysis objectives periodically when panel is open and thread is selected
  useEffect(() => {
    if (!isOpen || !currentThreadId) return;
    
    const refreshObjectives = async () => {
      try {
        const state = await getThreadState(currentThreadId);
        setAnalysisObjectives(state.analysis_objectives || []);
      } catch (err) {
        console.error('Failed to refresh analysis objectives:', err);
      }
    };
    
    // Refresh immediately when panel opens
    refreshObjectives();
    
    // Then refresh every 5 seconds while panel is open
    const interval = setInterval(refreshObjectives, 5000);
    
    return () => clearInterval(interval);
  }, [isOpen, currentThreadId]);

  /**
   * Save config: thread-specific or default depending on selection
   */
  async function handleSave() {
    setIsSaving(true);
    setSaveStatus('idle');

    try {
      if (currentThreadId) {
        // Save thread-specific config
        const updated = await updateThreadConfig(currentThreadId, config);
        setConfig(updated);
        // Update context circle immediately with new context_window
        if (updated.context_window !== null && updated.context_window !== undefined) {
          setContextUsage(contextUsage.tokensUsed, updated.context_window);
        }
      } else {
        // Save as default config for new threads
        setDefaultConfig({
          model: config.model,
          temperature: config.temperature,
          system_prompt: config.system_prompt,
          context_window: config.context_window,
        });
        // Update context circle with new default context_window
        if (config.context_window !== null && config.context_window !== undefined) {
          setContextUsage(contextUsage.tokensUsed, config.context_window);
        }
      }
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (err) {
      console.error('Failed to save config:', err);
      setSaveStatus('error');
    } finally {
      setIsSaving(false);
    }
  }

  /**
   * Save API keys
   */
  async function handleSaveApiKeys() {
    setIsSavingKeys(true);
    setKeysSaveStatus('idle');

    try {
      const keysToSave = {
        openai_key: apiKeyInputs.openai || null,
        anthropic_key: apiKeyInputs.anthropic || null,
      };
      
      const savedKeys = await saveUserApiKeys(userId, keysToSave);
      // Update store with masked keys
      setApiKeys({
        openai: savedKeys.openai_key || null,
        anthropic: savedKeys.anthropic_key || null,
      });
      setKeysSaveStatus('success');
      setTimeout(() => setKeysSaveStatus('idle'), 2000);
    } catch (err) {
      console.error('Failed to save API keys:', err);
      setKeysSaveStatus('error');
    } finally {
      setIsSavingKeys(false);
    }
  }

  if (!isOpen) return null;

  return (
    <aside className="w-80 bg-white dark:bg-slate-800 border-l border-gray-200 dark:border-slate-700 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-slate-700">
        <div className="flex items-center gap-2">
          <Settings size={18} className="text-gray-600 dark:text-slate-400" />
          <h2 className="font-semibold">
            {currentThreadId ? 'Thread Config' : 'Default Settings'}
          </h2>
        </div>
        <button
          onClick={toggleConfigPanel}
          className="p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
        >
          <X size={18} />
        </button>
      </div>

      {/* Config form */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {!currentThreadId && (
          <p className="text-xs text-gray-600 dark:text-slate-400 mb-4">
            These settings will be applied to new chats you create.
          </p>
        )}

        {/* Model selection */}
        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <select
            value={config.model || 'gpt-4.1'}
            onChange={(e) => setConfig({ ...config, model: e.target.value })}
            className="w-full px-3 py-2 rounded-lg focus:outline-none text-sm"
            style={{ 
              border: '1px solid var(--border)', 
              backgroundColor: 'var(--bg-secondary)', 
              color: 'var(--text-primary)'
            }}
          >
            <optgroup label="OpenAI">
              <option value="gpt-4.1">GPT-4.1</option>
              <option value="gpt-5">GPT-5</option>
            </optgroup>
            <optgroup label="Anthropic">
              <option value="claude-sonnet-4-5">Claude Sonnet 4.5</option>
              <option value="claude-haiku-4-5">Claude Haiku 4.5</option>
            </optgroup>
          </select>
        </div>

        {/* Temperature slider */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Temperature: {config.temperature?.toFixed(1) || '0.7'}
          </label>
          <input
            type="range"
            min="0.0"
            max="2.0"
            step="0.1"
            value={config.temperature ?? 0.7}
            onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
            className="w-full"
            style={{ 
              accentColor: 'var(--user-message-bg)',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '0.5rem',
              height: '6px'
            }}
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-slate-400 mt-1">
            <span>Precise (0.0)</span>
            <span>Creative (2.0)</span>
          </div>
        </div>

        {/* Context Window */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Context Window: {(config.context_window ?? defaultConfig.context_window ?? 30000).toLocaleString()} tokens
          </label>
          <input
            type="number"
            min="1000"
            max="200000"
            step="1000"
            value={config.context_window ?? defaultConfig.context_window ?? 30000}
            onChange={(e) => setConfig({ ...config, context_window: parseInt(e.target.value) || null })}
            className="w-full px-3 py-2 rounded-lg focus:outline-none text-sm"
            style={{ 
              border: '1px solid var(--border)', 
              backgroundColor: 'var(--bg-secondary)', 
              color: 'var(--text-primary)'
            }}
          />
          <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
            Maximum context size before summarization (GPT-4o: 128k, Base tier: 30k)
          </p>
        </div>

        {/* Custom Instructions (added to default prompt) */}
        <div>
          <label className="block text-sm font-medium mb-1">Custom Instructions</label>
          <textarea
            value={config.system_prompt || ''}
            onChange={(e) => setConfig({ ...config, system_prompt: e.target.value })}
            placeholder="Custom instructions for the assistant"
            rows={6}
            className="w-full px-3 py-2 rounded-lg focus:outline-none resize-none text-sm"
            style={{ 
              border: '1px solid var(--border)', 
              backgroundColor: 'var(--bg-secondary)', 
              color: 'var(--text-primary)'
            }}
          />
        </div>

        {/* Analysis Objectives (read-only, set by analyst) */}
        {currentThreadId && (
          <div className="border-t border-gray-200 dark:border-slate-700 pt-4">
            <div className="flex items-center gap-2 mb-2">
              <Target size={16} className="text-gray-600 dark:text-slate-400" />
              <label className="block text-sm font-medium">Analysis Objectives</label>
            </div>
            {analysisObjectives.length > 0 ? (
              <ul className="list-disc list-inside space-y-1 px-3 py-2 rounded-lg text-sm"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)', 
                  color: 'var(--text-primary)'
                }}
              >
                {analysisObjectives.map((objective, idx) => (
                  <li key={idx} className="text-xs">{objective}</li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-gray-500 dark:text-slate-400 italic px-3 py-2 rounded-lg"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)'
                }}
              >
                None set
              </p>
            )}
            <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
              Objectives are set automatically by the analyst during analysis
            </p>
          </div>
        )}

        {/* API Keys Section */}
        <div className="border-t border-gray-200 dark:border-slate-700 pt-4">
          <div className="flex items-center gap-2 mb-3">
            <Key size={16} className="text-gray-600 dark:text-slate-400" />
            <h3 className="text-sm font-medium">API Keys</h3>
          </div>
          
          <p className="text-xs text-gray-600 dark:text-slate-400 mb-4">
            Your API keys are encrypted and stored securely. They will be used for LLM requests.
          </p>

          {/* OpenAI API Key */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">OpenAI API Key</label>
            <div className="relative">
              <input
                type={showKeys.openai ? 'text' : 'password'}
                value={apiKeyInputs.openai}
                onChange={(e) => setApiKeyInputs({ ...apiKeyInputs, openai: e.target.value })}
                placeholder="sk-..."
                className="w-full px-3 py-2 pr-10 rounded-lg focus:outline-none text-sm"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)', 
                  color: 'var(--text-primary)'
                }}
              />
              <button
                type="button"
                onClick={() => setShowKeys({ ...showKeys, openai: !showKeys.openai })}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
              >
                {showKeys.openai ? (
                  <EyeOff size={16} className="text-gray-500" />
                ) : (
                  <Eye size={16} className="text-gray-500" />
                )}
              </button>
            </div>
          </div>

          {/* Anthropic API Key */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Anthropic API Key</label>
            <div className="relative">
              <input
                type={showKeys.anthropic ? 'text' : 'password'}
                value={apiKeyInputs.anthropic}
                onChange={(e) => setApiKeyInputs({ ...apiKeyInputs, anthropic: e.target.value })}
                placeholder="sk-ant-..."
                className="w-full px-3 py-2 pr-10 rounded-lg focus:outline-none text-sm"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)', 
                  color: 'var(--text-primary)'
                }}
              />
              <button
                type="button"
                onClick={() => setShowKeys({ ...showKeys, anthropic: !showKeys.anthropic })}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
              >
                {showKeys.anthropic ? (
                  <EyeOff size={16} className="text-gray-500" />
                ) : (
                  <Eye size={16} className="text-gray-500" />
                )}
              </button>
            </div>
          </div>

          {/* Save API Keys Button */}
          <button
            onClick={handleSaveApiKeys}
            disabled={isSavingKeys}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-white rounded-lg transition-colors text-sm"
            style={{ 
              backgroundColor: 'var(--user-message-bg)',
              opacity: isSavingKeys ? 0.6 : 1
            }}
          >
            <Key size={16} />
            <span>{isSavingKeys ? 'Saving...' : 'Save API Keys'}</span>
          </button>
          
          {/* API Keys save status feedback */}
          {keysSaveStatus === 'success' && (
            <p className="text-xs text-green-600 dark:text-green-400 mt-2 text-center">✓ API keys saved successfully</p>
          )}
          {keysSaveStatus === 'error' && (
            <p className="text-xs text-red-600 dark:text-red-400 mt-2 text-center">✗ Failed to save API keys</p>
          )}
        </div>
      </div>

      {/* Footer with save button */}
      <div className="p-4 border-t border-gray-200 dark:border-slate-700">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 text-white rounded-lg transition-colors"
          style={{ 
            backgroundColor: 'var(--user-message-bg)',
            opacity: isSaving ? 0.6 : 1
          }}
        >
          <Save size={18} />
          <span>{isSaving ? 'Saving...' : 'Save Config'}</span>
        </button>
        
        {/* Save status feedback */}
        {saveStatus === 'success' && (
          <p className="text-xs text-green-600 dark:text-green-400 mt-2 text-center">✓ Saved successfully</p>
        )}
        {saveStatus === 'error' && (
          <p className="text-xs text-red-600 dark:text-red-400 mt-2 text-center">✗ Failed to save</p>
        )}
      </div>
    </aside>
  );
}
