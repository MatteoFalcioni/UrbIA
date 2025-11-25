/**
 * CreativitySlider: Slider component for adjusting temperature/creativity.
 * Shows in the message input bubble below the context window slider.
 */

import { useState, useEffect, useRef } from 'react';
import { getThreadConfig, updateThreadConfig } from '@/utils/api';
import { useChatStore } from '@/store/chatStore';

const MIN_TEMP = 0;
const MAX_TEMP = 1;
const STEP = 0.1;

export function CreativitySlider() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const defaultConfig = useChatStore((state) => state.defaultConfig);
  const setDefaultConfig = useChatStore((state) => state.setDefaultConfig);
  
  const [temperature, setTemperature] = useState<number>(0.5);
  const [localValue, setLocalValue] = useState<number>(0.5); // For smooth visual feedback during drag
  const [isLoading, setIsLoading] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isDraggingRef = useRef<boolean>(false);

  // Load current temperature
  const loadTemperature = async () => {
    if (!currentThreadId) {
      // No thread: use default config
      const temp = defaultConfig.temperature ?? 0.5;
      setTemperature(temp);
      setLocalValue(temp);
      return;
    }

    try {
      const config = await getThreadConfig(currentThreadId);
      const temp = config.temperature ?? defaultConfig.temperature ?? 0.5;
      setTemperature(temp);
      setLocalValue(temp);
    } catch (err) {
      console.error('Failed to load thread config:', err);
      const temp = defaultConfig.temperature ?? 0.5;
      setTemperature(temp);
      setLocalValue(temp);
    }
  };

  // Load temperature when thread changes
  useEffect(() => {
    loadTemperature();
  }, [currentThreadId, defaultConfig.temperature]);

  // Handle mouse/touch release globally
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isDraggingRef.current && inputRef.current) {
        isDraggingRef.current = false;
        const value = parseFloat(inputRef.current.value);
        setLocalValue(value);
        handleChange(value);
      }
    };

    const handleGlobalTouchEnd = () => {
      if (isDraggingRef.current && inputRef.current) {
        isDraggingRef.current = false;
        const value = parseFloat(inputRef.current.value);
        setLocalValue(value);
        handleChange(value);
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
    if (value === temperature) return;

    setIsLoading(true);
    try {
      if (currentThreadId) {
        // Update thread config
        await updateThreadConfig(currentThreadId, { temperature: value });
        setTemperature(value);
        setLocalValue(value);
      } else {
        // No thread: update default config in store
        setDefaultConfig({
          ...defaultConfig,
          temperature: value,
        });
        setTemperature(value);
        setLocalValue(value);
      }
    } catch (err) {
      console.error('Failed to update temperature:', err);
      alert('Failed to update temperature');
    } finally {
      setIsLoading(false);
    }
  }

  const percentage = ((localValue - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)) * 100;

  return (
    <div ref={sliderRef} className="relative">
      {/* Slider container */}
      <div className="flex items-center gap-0.5 relative">
        <span className="text-[10px] font-semibold leading-none" style={{ color: 'var(--text-secondary)', lineHeight: '1' }}>
          Creativity
        </span>
        <div className="flex-1 relative flex items-center" style={{ minWidth: '80px', maxWidth: '140px' }}>
          <input
            ref={inputRef}
            type="range"
            min={MIN_TEMP}
            max={MAX_TEMP}
            step={STEP}
            value={localValue}
            onMouseDown={() => {
              isDraggingRef.current = true;
            }}
            onTouchStart={() => {
              isDraggingRef.current = true;
            }}
            onChange={(e) => {
              // During drag: update visual value smoothly without saving
              const value = parseFloat(e.target.value);
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
            minWidth: '24px',
            lineHeight: '1',
          }}
        >
          {localValue.toFixed(1)}
        </div>
      </div>
    </div>
  );
}

