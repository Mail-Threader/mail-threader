import type { ReactNode } from 'react';
import { DashboardLayout as ActualDashboardLayout } from '@/components/layout/dashboard-layout';
import { AuthGuard } from '@/components/auth/auth-guard';

export default function DashboardLayout({ children }: { children: ReactNode }) {
	return (
		<AuthGuard>
			<ActualDashboardLayout>{children}</ActualDashboardLayout>
		</AuthGuard>
	);
}
