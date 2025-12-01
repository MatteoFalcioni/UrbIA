import { useEffect, useState } from 'react';
import { ListTodo, ChevronDown, ChevronUp, CheckCircle2, Circle, Clock, XCircle } from 'lucide-react';
import { useChatStore } from '@/store/chatStore';
import { getThreadState } from '@/utils/api';
import type { Todo } from '@/types/api';

export function TodoListDropdown() {
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const todos = useChatStore((state) => state.todos);
  const setTodos = useChatStore((state) => state.setTodos);
  const [isExpanded, setIsExpanded] = useState(false);

  // Load todos once when thread changes
  useEffect(() => {
    if (!currentThreadId) {
      setTodos([]);
      return;
    }

    const loadTodos = async () => {
      try {
        const state = await getThreadState(currentThreadId);
        setTodos(state.todos || []);
      } catch (err: any) {
        if (err?.response?.status !== 404) {
          console.error('Failed to load todos:', err);
        }
        setTodos([]);
      }
    };

    loadTodos();
  }, [currentThreadId, setTodos]);

  if (!currentThreadId) return null;

  const getStatusIcon = (status: Todo['status']) => {
    switch (status) {
      case 'completed': return <CheckCircle2 size={14} className="text-green-500" />;
      case 'in_progress': return <Clock size={14} className="text-blue-500 animate-pulse" />;
      case 'cancelled': return <XCircle size={14} className="text-gray-400" />;
      default: return <Circle size={14} className="text-yellow-500" />; // pending
    }
  };
  
  const getStatusColor = (status: Todo['status']) => {
      switch (status) {
      case 'completed': return 'text-green-600 dark:text-green-400';
      case 'in_progress': return 'text-blue-600 dark:text-blue-400';
      case 'cancelled': return 'text-gray-500';
      default: return 'text-yellow-600 dark:text-yellow-400';
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all duration-200 hover:bg-gray-100 dark:hover:bg-slate-700"
        style={{ color: 'var(--text-primary)' }}
      >
        <ListTodo size={16} className="text-gray-600 dark:text-slate-400" />
        <span className="font-medium">Tasks</span>
        {todos.length > 0 ? (
          <span className="text-xs opacity-60">({todos.filter(t => t.status === 'completed').length}/{todos.length})</span>
        ) : (
          <span className="text-xs opacity-40 italic">none</span>
        )}
        {isExpanded ? <ChevronUp size={14} className="opacity-60" /> : <ChevronDown size={14} className="opacity-60" />}
      </button>

      {isExpanded && (
        <div 
          className="absolute top-full right-0 mt-1 w-96 rounded-lg shadow-lg border z-50 max-h-[80vh] overflow-y-auto"
          style={{ backgroundColor: 'var(--bg-primary)', borderColor: 'var(--border)' }}
        >
          <div className="p-3">
            <div className="flex items-center gap-2 mb-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <ListTodo size={14} className="text-gray-600 dark:text-slate-400" />
              <span className="text-xs font-medium opacity-75">Task List</span>
            </div>
            {todos.length > 0 ? (
              <ul className="space-y-2">
                {todos.map((todo, idx) => (
                  <li key={idx} className="flex gap-2 text-sm items-start p-1.5 rounded hover:bg-gray-50 dark:hover:bg-slate-800/50">
                    <div className="mt-0.5 flex-shrink-0">{getStatusIcon(todo.status)}</div>
                    <span className={`flex-1 ${todo.status === 'completed' ? 'text-gray-500 line-through' : 'text-gray-800 dark:text-gray-200'}`}>
                      {todo.content}
                    </span>
                    <span className={`text-[10px] uppercase tracking-wider font-semibold mt-0.5 ${getStatusColor(todo.status)}`}>
                        {todo.status.replace('_', ' ')}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs opacity-60 italic text-center py-2">No tasks yet</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

