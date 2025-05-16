import { Fragment, type ReactNode } from 'react';
// import {
// 	SidebarProvider,
// 	Sidebar,
// 	SidebarHeader,
// 	SidebarTrigger,
// 	SidebarContent,
// 	SidebarFooter,
// 	SidebarInset,
// 	SidebarRail,
// } from '@/components/ui/sidebar';
import { ThemeToggle } from '@/components/theme-toggle';
import { DashboardNav } from './dashboard-nav';
import { MailThreaderLogo } from '@/components/icons/mail-threader-logo';
import Link from 'next/link';
import { LogoutButton } from './logout-button';
import Sidebar from './sidebar';

interface DashboardLayoutProps {
	children: ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
	return (
		<Fragment>
			<Sidebar>{children}</Sidebar>
		</Fragment>
	);

	return (
		<SidebarProvider defaultOpen={true}>
			<Sidebar
				side="left"
				variant="sidebar"
				collapsible="icon"
				className="border-r"
			>
				<SidebarHeader className="flex items-center justify-between p-4">
					<Link
						href="/dashboard/data-view"
						className="flex items-center gap-2"
					>
						<MailThreaderLogo className="h-7 w-7 text-primary" />
						<h1 className="text-lg font-semibold tracking-tight group-data-[collapsible=icon]:hidden">
							Mail-Threader
						</h1>
					</Link>
					<SidebarTrigger className="group-data-[collapsible=icon]:hidden" />
				</SidebarHeader>
				<SidebarContent className="p-2">
					<DashboardNav />
				</SidebarContent>
				<SidebarFooter className="p-4 border-t">
					<div className="flex items-center justify-between group-data-[collapsible=icon]:justify-center">
						<ThemeToggle />
						<LogoutButton className="group-data-[collapsible=icon]:hidden ml-2 md:group-data-[collapsible=icon]:flex md:group-data-[collapsible=icon]:ml-0" />
					</div>
				</SidebarFooter>
			</Sidebar>
			<SidebarRail />
			<SidebarInset className="p-4 sm:p-6 lg:p-8">
				{children}
			</SidebarInset>
		</SidebarProvider>
	);
}
