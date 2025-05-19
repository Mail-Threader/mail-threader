import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import {
	Card,
	CardContent,
	CardHeader,
	CardTitle,
	CardDescription,
} from '@/components/ui/card';
import { MailThreaderLogo } from '@/components/icons/mail-threader-logo';
import {
	BarChartBigIcon,
	FileTextIcon,
	ChevronRightIcon,
	ShuffleIcon,
	GitMergeIcon,
} from 'lucide-react'; // TableIcon removed as unused

const features = [
	{
		icon: GitMergeIcon,
		title: 'Intelligent Email Threading',
		description:
			'Automatically group related emails into coherent threads, making it easy to follow conversations and understand context.',
		imageSrc: 'https://placehold.co/600x400.png',
		imageHint: 'email threads',
		link: '/dashboard/data-view',
	},
	{
		icon: FileTextIcon,
		title: 'AI-Powered Summarization',
		description:
			'Cut through the noise. Our AI condenses lengthy email threads and documents into concise, actionable summaries, highlighting crucial information.',
		imageSrc: 'https://placehold.co/600x400.png',
		imageHint: 'ai document',
		link: '/dashboard/summarization',
	},
	{
		icon: BarChartBigIcon,
		title: 'Insightful Visualizations',
		description:
			'Visualize communication patterns, topic clusters, and sentiment trends within your email data using interactive charts and graphs.',
		imageSrc: 'https://placehold.co/600x400.png',
		imageHint: 'charts graph',
		link: '/dashboard/visualizations',
	},
	{
		icon: ShuffleIcon,
		title: 'Advanced Data Processing',
		description:
			'Efficiently process and prepare large email datasets for analysis, with tools for cleaning, filtering, and structuring your data.',
		imageSrc: 'https://placehold.co/600x400.png',
		imageHint: 'data processing',
		link: '/dashboard/data-view',
	},
];

export default function LandingPage() {
	const currentYear = new Date().getFullYear();

	return (
		<div className="flex flex-col min-h-screen bg-background text-foreground">
			{/* Main Content */}
			<main className="flex-1">
				{/* Hero Section */}
				<section className="py-16 md:py-24 lg:py-32 bg-gradient-to-b from-background to-muted/50">
					<div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
						<h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
							<span className="block">Organize Your Emails</span>
							<span className="block text-primary">
								with AI-Powered Threading
							</span>
						</h1>
						<p className="mt-6 max-w-md mx-auto text-lg text-muted-foreground sm:text-xl md:mt-8 md:max-w-3xl">
							Mail-Threader helps you make sense of complex email
							datasets. Intelligently group conversations,
							summarize key points, and visualize communication
							patterns.
						</p>
						<div className="mt-10 max-w-sm mx-auto sm:max-w-none sm:flex sm:justify-center space-y-4 sm:space-y-0 sm:space-x-4">
							<Button
								size="lg"
								asChild
								className="w-full sm:w-auto"
							>
								<Link href="/signup">Get Started for Free</Link>
							</Button>
							<Button
								size="lg"
								variant="outline"
								asChild
								className="w-full sm:w-auto"
							>
								<Link href="#features">Explore Features</Link>
							</Button>
						</div>
					</div>
				</section>

				{/* Features Section */}
				<section id="features" className="py-16 md:py-24 bg-background">
					<div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
						<div className="text-center mb-12">
							<h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
								Powerful Tools for Email Analysis
							</h2>
							<p className="mt-4 text-lg text-muted-foreground">
								Discover insights from your email archives with
								Mail-Threader&apos;s intelligent features.
							</p>
						</div>
						<div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-2">
							{features.map((feature) => (
								<Card
									key={feature.title}
									className="overflow-hidden shadow-lg hover:shadow-xl transition-shadow duration-300 flex flex-col"
								>
									<CardHeader className="p-6">
										<div className="flex items-center gap-4 mb-2">
											<feature.icon className="h-10 w-10 text-primary" />
											<CardTitle className="text-2xl">
												{feature.title}
											</CardTitle>
										</div>
										<CardDescription className="text-base">
											{feature.description}
										</CardDescription>
									</CardHeader>
									<CardContent className="p-6 pt-0 flex-grow flex flex-col justify-between">
										<div className="aspect-video w-full relative rounded-md overflow-hidden mb-4">
											<Image
												src={feature.imageSrc}
												alt={feature.title}
												fill={true}
												style={{ objectFit: 'cover' }}
												data-ai-hint={feature.imageHint}
											/>
										</div>
										<Button
											variant="outline"
											asChild
											className="mt-auto w-full sm:w-auto self-start"
										>
											<Link href={feature.link}>
												Learn More{' '}
												<ChevronRightIcon className="ml-2 h-4 w-4" />
											</Link>
										</Button>
									</CardContent>
								</Card>
							))}
						</div>
					</div>
				</section>

				{/* How It Works Section */}
				<section className="py-16 md:py-24 bg-muted">
					<div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
						<div className="text-center">
							<h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
								Transforming Email Data into Knowledge
							</h2>
							<p className="mt-4 max-w-3xl mx-auto text-lg text-muted-foreground">
								Mail-Threader leverages cutting-edge AI and data
								processing techniques to transform raw email
								datasets into structured, analyzable resources.
								Our platform enables researchers, analysts, and
								legal professionals to explore email archives
								with unprecedented ease and depth.
							</p>
							<div className="mt-10">
								<Image
									src="https://placehold.co/1200x600.png"
									alt="Data processing pipeline"
									width={1200}
									height={600}
									className="rounded-lg shadow-xl mx-auto"
									data-ai-hint="data pipeline"
								/>
							</div>
						</div>
					</div>
				</section>

				{/* CTA Section */}
				<section className="py-16 md:py-24 bg-primary text-primary-foreground">
					<div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
						<h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
							Ready to Untangle Your Emails?
						</h2>
						<p className="mt-4 max-w-2xl mx-auto text-lg text-primary-foreground/80">
							Sign up today to access the full suite of
							Mail-Threader tools and start your journey into
							organized email analysis.
						</p>
						<div className="mt-10 max-w-sm mx-auto sm:max-w-none sm:flex sm:justify-center space-y-4 sm:space-y-0 sm:space-x-4">
							<Button
								size="lg"
								variant="secondary"
								asChild
								className="w-full sm:w-auto"
							>
								<Link href="/signup">Sign Up for Free</Link>
							</Button>
							<Button
								size="lg"
								variant="outline"
								asChild
								className="w-full sm:w-auto border-primary-foreground/50 text-primary-foreground hover:bg-primary-foreground/10"
							>
								<Link href="/login">Existing User? Login</Link>
							</Button>
						</div>
					</div>
				</section>
			</main>

			{/* Footer */}
			<footer className="py-8 border-t bg-background">
				<div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-muted-foreground">
					<div className="flex justify-center space-x-6 mb-4">
						<Link
							href="/about"
							className="text-sm hover:text-primary hover:underline"
						>
							About Us
						</Link>
						<Link
							href="/contact"
							className="text-sm hover:text-primary hover:underline"
						>
							Contact
						</Link>
						<Link
							href="/#features"
							className="text-sm hover:text-primary hover:underline"
						>
							Features
						</Link>
					</div>
					<p>
						&copy; {currentYear} Mail-Threader. All rights reserved.
					</p>
					<p className="text-sm mt-2">
						Unlocking understanding from complex email data.
					</p>
				</div>
			</footer>
		</div>
	);
}
