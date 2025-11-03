/**
 * Syncs Clerk authentication with the chat store.
 * Updates userId when user signs in/out.
 */

import { useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import { useChatStore } from '@/store/chatStore';

export function useClerkSync() {
  const { user, isLoaded } = useUser();
  const setUserId = useChatStore((state) => state.setUserId);

  useEffect(() => {
    if (!isLoaded) return;

    if (user) {
      // User is signed in - use Clerk user ID
      setUserId(user.id);
    } else {
      // User is signed out - use or generate anonymous ID
      const stored = localStorage.getItem('userId');
      if (stored) {
        setUserId(stored);
      } else {
        const newId = crypto.randomUUID();
        localStorage.setItem('userId', newId);
        setUserId(newId);
      }
    }
  }, [user, isLoaded, setUserId]);
}

