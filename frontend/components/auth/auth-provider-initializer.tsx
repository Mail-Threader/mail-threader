'use client';

import { ReactNode, useEffect } from 'react';
import { useAuthStore } from '@/store/auth-store';

export function AuthProviderInitializer({ children }: { children: ReactNode }) {
	useEffect(() => {
		// Call checkSession only once on initial mount
		useAuthStore.getState().checkSession();
	}, []);

	return <>{children}</>;
}
