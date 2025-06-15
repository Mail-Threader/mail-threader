'use client';

import Link from 'next/link';
import React, { Fragment, useEffect, useState } from 'react';
import { MailThreaderLogo } from '../icons/mail-threader-logo';
import { Button } from '../ui/button';
import { useAuthStore } from '@/store/auth-store';
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar';
import { Skeleton } from '../ui/skeleton';
import {
	BarChartBigIcon,
	BookOpenTextIcon,
	FileTextIcon,
	LogOutIcon,
	Moon,
	Settings,
	Sun,
	TableIcon,
	UploadCloudIcon,
} from 'lucide-react';
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuGroup,
	DropdownMenuItem,
	DropdownMenuPortal,
	DropdownMenuSeparator,
	DropdownMenuSub,
	DropdownMenuSubContent,
	DropdownMenuSubTrigger,
	DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { usePreferencesStore } from '@/store/preferences-store';
import { useRouter } from 'next/navigation';
import { useToast } from '@/hooks/use-toast';
import { logoutAction } from '@/actions/auth';

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

const Nav = () => {
	const { isAuthenticated, user, checkSession, isLoading, logout } =
		useAuthStore();
	const currentTheme = usePreferencesStore((state) => state.theme);
	const setThemeInStore = usePreferencesStore((state) => state.setTheme);
	const [mounted, setMounted] = useState(false);
	const router = useRouter();
	const { toast } = useToast();

	useEffect(() => {
		checkSession();
		setMounted(true);
	}, []);

	useEffect(() => {
		if (!mounted) return; // Wait for store to be hydrated

		const root = window.document.documentElement;
		root.classList.remove('theme-light', 'dark');

		if (currentTheme === 'system') {
			const systemTheme = window.matchMedia(
				'(prefers-color-scheme: dark)',
			).matches
				? 'dark'
				: 'theme-light';
			root.classList.add(systemTheme);
		} else {
			root.classList.add(currentTheme);
		}
	}, [currentTheme, mounted]);

	useEffect(() => {
		if (!mounted || currentTheme !== 'system') {
			return;
		}

		const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
		const handleChange = () => {
			// This ensures the DOM updates if the system theme changes while 'system' is selected in the app
			const root = window.document.documentElement;
			root.classList.remove('theme-light', 'dark');
			const systemTheme = mediaQuery.matches ? 'dark' : 'theme-light';
			root.classList.add(systemTheme);
		};

		mediaQuery.addEventListener('change', handleChange);
		return () => mediaQuery.removeEventListener('change', handleChange);
	}, [currentTheme, mounted]);

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

	return (
		<header className="w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
			<div className="container flex h-16 items-center max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
				<Link
					href={isAuthenticated ? '/dashboard/data-view' : '/'}
					className="flex items-center gap-2 mr-6"
				>
					<MailThreaderLogo className="h-7 w-7 text-primary" />
					<span className="text-xl font-semibold">Mail-Threader</span>
				</Link>
				<nav className="flex items-center gap-1 sm:gap-4 text-sm font-medium ml-auto">
					{isLoading ? (
						<Fragment>
							<Skeleton className="h-6 w-6" />
							<Skeleton className="h-6 w-12" />
							<Skeleton className="h-6 w-6" />
							<Button variant="ghost" size="icon" disabled>
								<Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
								<Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
								<span className="sr-only">
									Toggle theme (loading)
								</span>
							</Button>
						</Fragment>
					) : isAuthenticated && user ? (
						<Fragment>
							<DropdownMenu>
								<DropdownMenuTrigger asChild>
									<Button variant="outline">View-Mode</Button>
								</DropdownMenuTrigger>
								<DropdownMenuContent
									className="w-56"
									align="start"
								>
									<DropdownMenuGroup>
										{navItems.map((item, i) => (
											<DropdownMenuItem key={i} asChild>
												<Link
													href={item.href}
													className="flex items-center gap-2"
												>
													<item.icon className="h-4 w-4" />
													{item.label}
												</Link>
											</DropdownMenuItem>
										))}
									</DropdownMenuGroup>
								</DropdownMenuContent>
							</DropdownMenu>
							<DropdownMenu>
								<DropdownMenuTrigger asChild>
									<Button variant="outline">
										<Settings />
									</Button>
								</DropdownMenuTrigger>
								<DropdownMenuContent
									className="w-56"
									align="start"
								>
									<DropdownMenuGroup></DropdownMenuGroup>
									<DropdownMenuSeparator />
									<DropdownMenuGroup>
										<DropdownMenuSub>
											<DropdownMenuSubTrigger>
												Toggle Theme
											</DropdownMenuSubTrigger>
											<DropdownMenuPortal>
												<DropdownMenuSubContent>
													<DropdownMenuItem
														onClick={() =>
															setThemeInStore(
																'theme-light',
															)
														}
													>
														Light
													</DropdownMenuItem>
													<DropdownMenuItem
														onClick={() =>
															setThemeInStore(
																'dark',
															)
														}
													>
														Dark
													</DropdownMenuItem>
													<DropdownMenuItem
														onClick={() =>
															setThemeInStore(
																'system',
															)
														}
													>
														System
													</DropdownMenuItem>
												</DropdownMenuSubContent>
											</DropdownMenuPortal>
										</DropdownMenuSub>
									</DropdownMenuGroup>
									<DropdownMenuSeparator />
									<DropdownMenuGroup>
										<DropdownMenuItem
											onClick={handleLogout}
										>
											<LogOutIcon />
											Signout
											<span className="sr-only">
												Logout
											</span>
										</DropdownMenuItem>
									</DropdownMenuGroup>
								</DropdownMenuContent>
							</DropdownMenu>
							<Link href="/dashboard/account">
								<Avatar>
									<AvatarImage src="https://github.com/shadcn.png" />
									<AvatarFallback>
										{user.email[0].toUpperCase()}
									</AvatarFallback>
								</Avatar>
							</Link>
						</Fragment>
					) : (
						<Fragment>
							<Button variant="ghost" asChild>
								<Link href="/about">About</Link>
							</Button>
							<Button variant="ghost" asChild>
								<Link href="/contact">Contact</Link>
							</Button>
							<Button variant="ghost" asChild>
								<Link href="/login">Login</Link>
							</Button>
							<Button asChild>
								<Link href="/signup">Sign Up</Link>
							</Button>
						</Fragment>
					)}
				</nav>
			</div>
		</header>
	);
};

export default Nav;
