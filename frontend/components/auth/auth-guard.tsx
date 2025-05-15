'use client';

import { useAuthStore } from '@/store/auth-store';
import { useRouter, usePathname } from 'next/navigation';
import type { ReactNode } from 'react';
import { useEffect } from 'react';
import { Skeleton } from '@/components/ui/skeleton';

interface AuthGuardProps {
	children: ReactNode;
}

export function AuthGuard({ children }: AuthGuardProps) {
	const { isAuthenticated, isLoading } = useAuthStore();
	const router = useRouter();
	const pathname = usePathname();

	useEffect(() => {
		// Wait for session check to complete
		if (!isLoading) {
			if (!isAuthenticated && pathname.startsWith('/dashboard')) {
				router.push('/login');
			}
		}
	}, [isAuthenticated, isLoading, router, pathname]);

	if (isLoading && pathname.startsWith('/dashboard')) {
		// Show a loading state while verifying auth
		return (
			<div className="flex flex-col items-center justify-center min-h-screen p-4">
				<Skeleton className="h-12 w-12 rounded-full mb-4" />
				<Skeleton className="h-4 w-[250px] mb-2" />
				<Skeleton className="h-4 w-[200px]" />
			</div>
		);
	}

	// If not loading and not authenticated on a dashboard route, redirect should have happened or is happening.
	// Return null to prevent rendering children if redirect is underway.
	if (!isLoading && !isAuthenticated && pathname.startsWith('/dashboard')) {
		return null;
	}

	return <>{children}</>;
}
