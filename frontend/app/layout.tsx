import type { Metadata } from 'next';
import './globals.css';
import { Toaster } from '@/components/ui/toaster';
import { ReactNode } from 'react';
import Nav from '@/components/layout/nav';
import { NuqsAdapter } from 'nuqs/adapters/next/app';
import { TooltipProvider } from '@/components/ui/tooltip';

export const metadata: Metadata = {
	title: 'Mail-Threader',
	description: 'Analytics and Threading for Email Datasets',
};

export default function RootLayout({
	children,
}: Readonly<{
	children: ReactNode;
}>) {
	return (
		<html lang="en" suppressHydrationWarning>
			<body className={`antialiased font-sans`}>
				<NuqsAdapter>
					<TooltipProvider>
						<Nav />
						{children}
						<Toaster />
					</TooltipProvider>
				</NuqsAdapter>
			</body>
		</html>
	);
}
