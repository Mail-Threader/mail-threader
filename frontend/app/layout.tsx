import type { Metadata } from 'next';
import './globals.css';
import { Toaster } from '@/components/ui/toaster';
import { ReactNode } from 'react';

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
				{children}
				<Toaster />
			</body>
		</html>
	);
}
