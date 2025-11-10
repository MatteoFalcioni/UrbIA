/**
 * SettingsPage: Full page for thread-specific or default configuration.
 * - When thread is selected: shows and saves thread-specific config
 * - When no thread: shows and saves default config for new threads
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Save, Settings, Eye, EyeOff, Key } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { getDefaultConfig, getThreadConfig, updateThreadConfig, getUserApiKeys, saveUserApiKeys } from '@/utils/api';
import type { ThreadConfig } from '@/types/api';

export function SettingsPage() {
  const navigate = useNavigate();
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const userId = useChatStore((state) => state.userId);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  const setDefaultConfig = useChatStore((state) => state.setDefaultConfig);
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

      // Thread selected: load thread-specific config
      try {
        const cfg = await getThreadConfig(currentThreadId!);
        setConfig(cfg);
      } catch (err) {
        console.error('Failed to load config:', err);
      }
    }
    loadConfig();
  }, [currentThreadId]);

  // Load API keys on mount
  useEffect(() => {
    async function loadApiKeys() {
      try {
        const keys = await getUserApiKeys(userId);
        // Don't set masked keys in inputs - they're just for display/checking existence
        // Leave inputs empty so user can enter new keys
        // Only show placeholder if keys exist
        setApiKeyInputs({
          openai: '',
          anthropic: '',
        });
        // Update store to track if keys exist (for warning modal)
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

  /**
   * Save config: thread-specific or default depending on selection
   */
  async function handleSave() {
    setIsSaving(true);
    setSaveStatus('idle');
    try {
      if (!currentThreadId) {
        // Save as default config (PUT /config/defaults)
        await updateThreadConfig('', config);
        // Also update store
        setDefaultConfig({
          model: config.model,
          temperature: config.temperature,
          system_prompt: config.system_prompt,
          context_window: config.context_window,
        });
      } else {
        // Save thread-specific config (PUT /threads/:id/config)
        await updateThreadConfig(currentThreadId, config);
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
      // Only send keys that have been entered (non-empty)
      const keysToSave: { openai_key?: string | null; anthropic_key?: string | null } = {};
      if (apiKeyInputs.openai) {
        keysToSave.openai_key = apiKeyInputs.openai;
      }
      if (apiKeyInputs.anthropic) {
        keysToSave.anthropic_key = apiKeyInputs.anthropic;
      }
      
      await saveUserApiKeys(userId, keysToSave);
      
      // Update store with the keys that were saved
      const updates: { openai?: string | null; anthropic?: string | null } = {};
      if (apiKeyInputs.openai) {
        updates.openai = apiKeyInputs.openai;
      }
      if (apiKeyInputs.anthropic) {
        updates.anthropic = apiKeyInputs.anthropic;
      }
      setApiKeys(updates);
      
      setKeysSaveStatus('success');
      
      // Redirect to chat after successful save (so warning disappears)
      setTimeout(() => {
        navigate('/');
      }, 1500);
    } catch (err) {
      console.error('Failed to save API keys:', err);
      setKeysSaveStatus('error');
    } finally {
      setIsSavingKeys(false);
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-slate-900 text-gray-700 dark:text-slate-200 overflow-hidden">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-gray-200 dark:border-slate-700 px-6 py-4 flex-shrink-0">
        <div className="max-w-4xl mx-auto flex items-center gap-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
            title="Back to chat"
          >
            <ArrowLeft size={20} />
          </button>
          <div className="flex items-center gap-3 flex-1">
            <Settings size={24} className="text-gray-600 dark:text-slate-400" />
            <div>
              <h1 className="text-xl font-semibold">Settings</h1>
              <p className="text-sm text-gray-500 dark:text-slate-400">
                {currentThreadId ? 'Thread-specific configuration' : 'Default configuration for new threads'}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main content - scrollable area */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto p-6 space-y-6">
        {/* Model Configuration */}
        <section className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Settings size={18} className="text-gray-600 dark:text-slate-400" />
            Model Configuration
          </h2>

          <div className="space-y-4">
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">Model</label>
              <select
                value={config.model || defaultConfig.model || 'gpt-4.1'}
                onChange={(e) => setConfig((prev) => ({ ...prev, model: e.target.value }))}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-800 dark:focus:ring-gray-600 focus:border-gray-800 dark:focus:border-gray-600 transition-all outline-none"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)', 
                  color: 'var(--text-primary)'
                }}
              >
                <option value="gpt-4.1">GPT-4.1</option>
                <option value="claude-3-5-sonnet-20241022">Claude Sonnet 4.5</option>
                <option value="claude-3-5-haiku-20241022">Claude Haiku 4.5</option>
              </select>
            </div>

            {/* Temperature */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Temperature: {config.temperature ?? defaultConfig.temperature ?? 0.5}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={config.temperature ?? defaultConfig.temperature ?? 0.5}
                onChange={(e) => setConfig((prev) => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                className="w-full h-2 rounded-lg cursor-pointer range-slider"
                style={{ 
                  background: `linear-gradient(to right, #1f2937 0%, #1f2937 ${((config.temperature ?? defaultConfig.temperature ?? 0.5) / 2) * 100}%, #e5e7eb ${((config.temperature ?? defaultConfig.temperature ?? 0.5) / 2) * 100}%, #e5e7eb 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-slate-400 mt-1">
                <span>Precise (0.0)</span>
                <span>Creative (2.0)</span>
              </div>
            </div>

            {/* Context Window */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Context Window: {(config.context_window ?? defaultConfig.context_window ?? 64000).toLocaleString()} tokens
              </label>
              <input
                type="number"
                min="1000"
                max="200000"
                step="1000"
                value={config.context_window ?? defaultConfig.context_window ?? 64000}
                onChange={(e) => setConfig((prev) => ({ ...prev, context_window: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-800 dark:focus:ring-gray-600 focus:border-gray-800 dark:focus:border-gray-600 transition-all outline-none"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)', 
                  color: 'var(--text-primary)'
                }}
              />
              <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
                Maximum context size before summarization (default: 64k context window)
              </p>
              {(config.context_window ?? defaultConfig.context_window ?? 64000) > 64000 && (
                <div className="mt-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                  <p className="text-xs text-yellow-800 dark:text-yellow-200">
                    ⚠️ <strong>Note:</strong> Context windows larger than 64k require your OpenAI and Anthropic accounts to be enabled for extended context. If not enabled, API calls may fail with an error.
                  </p>
                </div>
              )}
            </div>

            {/* System Prompt */}
            <div>
              <label className="block text-sm font-medium mb-2">System Prompt</label>
              <textarea
                value={config.system_prompt ?? defaultConfig.system_prompt ?? ''}
                onChange={(e) => setConfig((prev) => ({ ...prev, system_prompt: e.target.value }))}
                rows={6}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-800 dark:focus:ring-gray-600 focus:border-gray-800 dark:focus:border-gray-600 transition-all resize-none outline-none"
                style={{ 
                  border: '1px solid var(--border)', 
                  backgroundColor: 'var(--bg-secondary)', 
                  color: 'var(--text-primary)'
                }}
              />
            </div>

            {/* Save Button */}
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-900 dark:bg-gray-700 dark:hover:bg-gray-600 text-white rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Save size={16} />
              {isSaving ? 'Saving...' : 'Save Configuration'}
            </button>
            
            {saveStatus === 'success' && (
              <p className="text-sm text-green-600 dark:text-green-400 text-center">
                ✓ Configuration saved successfully
              </p>
            )}
            {saveStatus === 'error' && (
              <p className="text-sm text-red-600 dark:text-red-400 text-center">
                ✗ Failed to save configuration
              </p>
            )}
          </div>
        </section>

        {/* API Keys Section */}
        <section className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Key size={18} className="text-gray-600 dark:text-slate-400" />
            API Keys
          </h2>

          <div className="space-y-4">
            <p className="text-sm text-gray-600 dark:text-slate-400">
              API keys are encrypted and stored securely. Enter a new key to update it, or leave empty to keep the existing one.
            </p>

            {/* OpenAI Key */}
            <div>
              <label className="block text-sm font-medium mb-2">OpenAI API Key</label>
              <div className="relative">
                <input
                  type={showKeys.openai ? 'text' : 'password'}
                  value={apiKeyInputs.openai}
                  onChange={(e) => setApiKeyInputs((prev) => ({ ...prev, openai: e.target.value }))}
                  placeholder="sk-proj-... (enter new key or leave empty to keep existing)"
                  className="w-full px-3 py-2 pr-10 border rounded-lg focus:ring-2 focus:ring-gray-800 dark:focus:ring-gray-600 focus:border-gray-800 dark:focus:border-gray-600 transition-all outline-none"
                  style={{ 
                    border: '1px solid var(--border)', 
                    backgroundColor: 'var(--bg-secondary)', 
                    color: 'var(--text-primary)'
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowKeys((prev) => ({ ...prev, openai: !prev.openai }))}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
                >
                  {showKeys.openai ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Anthropic Key */}
            <div>
              <label className="block text-sm font-medium mb-2">Anthropic API Key</label>
              <div className="relative">
                <input
                  type={showKeys.anthropic ? 'text' : 'password'}
                  value={apiKeyInputs.anthropic}
                  onChange={(e) => setApiKeyInputs((prev) => ({ ...prev, anthropic: e.target.value }))}
                  placeholder="sk-ant-... (enter new key or leave empty to keep existing)"
                  className="w-full px-3 py-2 pr-10 border rounded-lg focus:ring-2 focus:ring-gray-800 dark:focus:ring-gray-600 focus:border-gray-800 dark:focus:border-gray-600 transition-all outline-none"
                  style={{ 
                    border: '1px solid var(--border)', 
                    backgroundColor: 'var(--bg-secondary)', 
                    color: 'var(--text-primary)'
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowKeys((prev) => ({ ...prev, anthropic: !prev.anthropic }))}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
                >
                  {showKeys.anthropic ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Save Keys Button */}
            <button
              onClick={handleSaveApiKeys}
              disabled={isSavingKeys}
              className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Save size={16} />
              {isSavingKeys ? 'Saving...' : 'Save API Keys'}
            </button>
            
            {keysSaveStatus === 'success' && (
              <p className="text-sm text-green-600 dark:text-green-400 text-center">
                ✓ API keys saved successfully
              </p>
            )}
            {keysSaveStatus === 'error' && (
              <p className="text-sm text-red-600 dark:text-red-400 text-center">
                ✗ Failed to save API keys
              </p>
            )}
          </div>
        </section>
        </div>
      </main>
    </div>
  );
}

