'use client';

import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import {
	SidebarProvider,
	Sidebar,
	SidebarHeader,
	SidebarTrigger,
	SidebarContent,
	SidebarInset,
	SidebarRail,
} from '@/components/ui/sidebar';
import { DashboardNav } from './dashboard-nav';
import { GlobalFilterControls } from './global-filter-controls'; // Import GlobalFilterControls

interface DashboardLayoutProps {
	children: ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
	const pathname = usePathname(); // Get current path
	const showGlobalFilters = pathname !== '/dashboard/account'; // Condition to show filters

	return (
		<SidebarProvider defaultOpen={false}>
			<Sidebar
				side="left"
				variant="sidebar"
				collapsible="icon"
				className="border-r"
			>
				<SidebarHeader className="flex justify-center items-center">
					<SidebarTrigger />
				</SidebarHeader>
				<SidebarContent className="p-2 mt-10">
					<DashboardNav />
				</SidebarContent>
			</Sidebar>
			<SidebarRail />
			<SidebarInset className="p-4 sm:p-6 lg:p-8">
				{showGlobalFilters && <GlobalFilterControls />}
				{children}
			</SidebarInset>
		</SidebarProvider>
	);
}
