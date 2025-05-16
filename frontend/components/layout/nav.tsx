'use client';

import Link from 'next/link';
import React, { Fragment, useEffect } from 'react';
import { MailThreaderLogo } from '../icons/mail-threader-logo';
import { Button } from '../ui/button';
import { useAuthStore } from '@/store/auth-store';
import { ThemeToggle } from '../theme-toggle';
import { LogoutButton } from './logout-button';
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar';

const Nav = () => {
	const { isAuthenticated, user, checkSession } = useAuthStore();

	useEffect(() => {
		checkSession();
	}, []);

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
				<nav className="flex items-center gap-2 sm:gap-4 text-sm font-medium ml-auto">
					{isAuthenticated && user ? (
						<Fragment>
							<ThemeToggle />
							<LogoutButton className="group-data-[collapsible=icon]:hidden ml-2 md:group-data-[collapsible=icon]:flex md:group-data-[collapsible=icon]:ml-0" />
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
