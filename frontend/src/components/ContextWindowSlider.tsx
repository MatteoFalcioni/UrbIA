/**
 * ContextWindowSlider: Slider component for adjusting context window size.
 * Shows in the message input bubble with warning when > 64k.
 */

import { useState, useEffect, useRef } from 'react';
import { getThreadConfig, updateThreadConfig } from '@/utils/api';
import { useChatStore } from '@/store/chatStore';

const MIN_CONTEXT = 16000;
const MAX_CONTEXT = 128000;
const STEP = 2000;

// Generate array of valid values (multiples of 2000)
const CONTEXT_VALUES = Array.from(
  { length: (MAX_CONTEXT - MIN_CONTEXT) / STEP + 1 },
  (_, i) => MIN_CONTEXT + i * STEP
);

export function ContextWindowSlider() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  const setDefaultConfig = useChatStore((state) => state.setDefaultConfig);
  
  const [contextWindow, setContextWindow] = useState<number>(64000);
  const [localValue, setLocalValue] = useState<number>(64000); // For smooth visual feedback during drag
  const [isLoading, setIsLoading] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isDraggingRef = useRef<boolean>(false);

  // Load current context window
  const loadContextWindow = async () => {
    if (!currentThreadId) {
      // No thread: use default config
      const ctx = defaultConfig.context_window ?? 64000;
      // Round to nearest valid value
      const rounded = CONTEXT_VALUES.reduce((prev, curr) => 
        Math.abs(curr - ctx) < Math.abs(prev - ctx) ? curr : prev
      );
      setContextWindow(rounded);
      setLocalValue(rounded);
      return;
    }

    try {
      const config = await getThreadConfig(currentThreadId);
      const ctx = config.context_window ?? defaultConfig.context_window ?? 64000;
      // Round to nearest valid value
      const rounded = CONTEXT_VALUES.reduce((prev, curr) => 
        Math.abs(curr - ctx) < Math.abs(prev - ctx) ? curr : prev
      );
      setContextWindow(rounded);
      setLocalValue(rounded);
    } catch (err) {
      console.error('Failed to load thread config:', err);
      const ctx = defaultConfig.context_window ?? 64000;
      const rounded = CONTEXT_VALUES.reduce((prev, curr) => 
        Math.abs(curr - ctx) < Math.abs(prev - ctx) ? curr : prev
      );
      setContextWindow(rounded);
      setLocalValue(rounded);
    }
  };

  // Load context window when thread changes
  useEffect(() => {
    loadContextWindow();
  }, [currentThreadId, defaultConfig.context_window]);

  const roundToStep = (value: number): number => {
    return CONTEXT_VALUES.reduce((prev, curr) => 
      Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
    );
  };

  // Handle mouse/touch release globally
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isDraggingRef.current && inputRef.current) {
        isDraggingRef.current = false;
        const value = parseInt(inputRef.current.value);
        const rounded = roundToStep(value);
        setLocalValue(rounded);
        handleChange(rounded);
      }
    };

    const handleGlobalTouchEnd = () => {
      if (isDraggingRef.current && inputRef.current) {
        isDraggingRef.current = false;
        const value = parseInt(inputRef.current.value);
        const rounded = roundToStep(value);
        setLocalValue(rounded);
        handleChange(rounded);
      }
    };

    document.addEventListener('mouseup', handleGlobalMouseUp);
    document.addEventListener('touchend', handleGlobalTouchEnd);

    return () => {
      document.removeEventListener('mouseup', handleGlobalMouseUp);
      document.removeEventListener('touchend', handleGlobalTouchEnd);
    };
  }, []);

  async function handleChange(value: number) {
    if (value === contextWindow) return;

    setIsLoading(true);
    try {
      if (currentThreadId) {
        // Update thread config
        await updateThreadConfig(currentThreadId, { context_window: value });
        setContextWindow(value);
        setLocalValue(value);
      } else {
        // No thread: update default config in store
        setDefaultConfig({
          ...defaultConfig,
          context_window: value,
        });
        setContextWindow(value);
        setLocalValue(value);
      }
    } catch (err) {
      console.error('Failed to update context window:', err);
      alert('Failed to update context window');
    } finally {
      setIsLoading(false);
    }
  }


  const percentage = ((localValue - MIN_CONTEXT) / (MAX_CONTEXT - MIN_CONTEXT)) * 100;
  const showWarningOverlay = localValue > 64000;

  return (
    <div ref={sliderRef} className="relative">
      {/* Warning overlay when > 64k - positioned well above the slider */}
      {showWarningOverlay && (
        <div
          className="absolute -top-20 left-1/2 -translate-x-1/2 p-2 rounded text-xs pointer-events-none z-50 transition-opacity duration-200"
          style={{
            backgroundColor: 'color-mix(in srgb, var(--bg-secondary) 30%, transparent)',
            backdropFilter: 'blur(8px)',
            border: '1px solid color-mix(in srgb, var(--border) 25%, transparent)',
            color: 'var(--text-primary)',
            opacity: 0.4,
            minWidth: '280px',
            maxWidth: '320px',
          }}
        >
          <p className="text-center leading-tight">
            ⚠️ Context windows larger than 64k tokens may not be supported by all API plans
          </p>
        </div>
      )}
      
      {/* Slider container */}
      <div className="flex items-center gap-0.5 relative">
        <span className="text-[10px] font-semibold leading-none" style={{ color: 'var(--text-secondary)', lineHeight: '1' }}>
          Context
        </span>
        <div className="flex-1 relative flex items-center" style={{ minWidth: '80px', maxWidth: '140px' }}>
          <input
            ref={inputRef}
            type="range"
            min={MIN_CONTEXT}
            max={MAX_CONTEXT}
            step="1000"
            value={localValue}
            onMouseDown={() => {
              isDraggingRef.current = true;
            }}
            onTouchStart={() => {
              isDraggingRef.current = true;
            }}
            onChange={(e) => {
              // During drag: update visual value smoothly without rounding or saving
              const value = parseInt(e.target.value);
              setLocalValue(value);
            }}
            disabled={isLoading}
            className="w-full h-1 rounded-lg cursor-pointer range-slider"
            style={{ 
              background: `linear-gradient(to right, rgb(31, 41, 55) 0%, rgb(31, 41, 55) ${percentage}%, rgb(229, 231, 235) ${percentage}%, rgb(229, 231, 235) 100%)`
            }}
          />
        </div>
        {/* Number to the right */}
        <div
          className="text-[10px] font-bold whitespace-nowrap leading-none"
          style={{
            color: 'var(--text-secondary)',
            minWidth: '28px',
            lineHeight: '1',
          }}
        >
          {localValue >= 1000 ? `${(localValue / 1000).toFixed(0)}k` : localValue}
        </div>
      </div>
    </div>
  );
}
