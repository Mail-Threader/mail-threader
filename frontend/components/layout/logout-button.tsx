'use client';

import { useAuthStore } from '@/store/auth-store';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { LogOutIcon } from 'lucide-react';
import { logoutAction } from '@/actions/auth';
import { useToast } from '@/hooks/use-toast';

export function LogoutButton({ className }: { className?: string }) {
	const { logout, isAuthenticated } = useAuthStore();
	const router = useRouter();
	const { toast } = useToast();

	const handleLogout = async () => {
		const result = await logoutAction();
		if (result.success) {
			logout(); // Update client-side store
			toast({
				title: 'Logged Out',
				description: 'You have been successfully logged out.',
			});
			router.push('/login');
		} else {
			toast({
				title: 'Logout Failed',
				description: 'Could not log out. Please try again.',
				variant: 'destructive',
			});
		}
	};

	if (!isAuthenticated) {
		return null;
	}

	return (
		<Button
			variant="ghost"
			onClick={handleLogout}
			className={className}
			aria-label="Logout"
		>
			<LogOutIcon />
			Signout
			<span className="sr-only">Logout</span>
		</Button>
	);
}
