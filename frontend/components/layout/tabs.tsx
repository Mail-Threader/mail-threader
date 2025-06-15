import { Button } from '@/components/ui/button';
import {
	BarChartBigIcon,
	BookOpenTextIcon,
	FileTextIcon,
	TableIcon,
	UploadCloudIcon,
	UserCogIcon,
} from 'lucide-react';
import Link from 'next/link';
import { ReactNode } from 'react';

const Tabs = ({}: Readonly<{
	children: ReactNode;
}>) => {
	const navItems = [
		{ href: '/dashboard/data-view', label: 'Data View', icon: TableIcon },
		{
			href: '/dashboard/upload-data',
			label: 'Upload Data',
			icon: UploadCloudIcon,
		},
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
			href: '/dashboard/account',
			label: 'Account Settings',
			icon: UserCogIcon,
		},
	];

	return (
		<div className="p-10 flex justify-center items-center gap-4">
			{navItems.map((navItem, i) => (
				<Button asChild key={i}>
					<Link href={navItem.href}></Link>
				</Button>
			))}
		</div>
	);
};

export default Tabs;
