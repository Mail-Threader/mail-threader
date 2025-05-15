'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
	SidebarMenu,
	SidebarMenuItem,
	SidebarMenuButton,
} from '@/components/ui/sidebar';
import {
	TableIcon,
	FileTextIcon,
	BarChartBigIcon,
	BookOpenTextIcon,
	UploadCloudIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
	{ href: '/dashboard/data-view', label: 'Data View', icon: TableIcon },

	{
		href: '/dashboard/summarization',
		label: 'Summarization',
		icon: FileTextIcon,
	},
	{
		href: '/dashboard/visualizations',
		label: 'Visualizations',
		icon: BarChartBigIcon,
	},
	{
		href: '/dashboard/stories',
		label: 'Story Explorer',
		icon: BookOpenTextIcon,
	},
	{
		href: '/dashboard/upload-data',
		label: 'Upload Data',
		icon: UploadCloudIcon,
	},
];

export function DashboardNav() {
	const pathname = usePathname();

	return (
		<SidebarMenu>
			{navItems.map((item) => (
				<SidebarMenuItem key={item.href}>
					<Link href={item.href} passHref>
						<SidebarMenuButton
							asChild
							isActive={
								pathname === item.href ||
								(pathname.startsWith(item.href) &&
									item.href !== '/dashboard')
							}
							tooltip={{
								children: item.label,
								side: 'right',
								align: 'center',
							}}
							className={cn(
								'w-full justify-start',
								pathname === item.href ||
									(pathname.startsWith(item.href) &&
										item.href !== '/dashboard')
									? 'bg-sidebar-accent text-sidebar-accent-foreground'
									: 'hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
							)}
						>
							<div className="flex items-center gap-2">
								<item.icon className="h-5 w-5" />
								<span>{item.label}</span>
							</div>
						</SidebarMenuButton>
					</Link>
				</SidebarMenuItem>
			))}
		</SidebarMenu>
	);
}
