import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { MailThreaderLogo } from '@/components/icons/mail-threader-logo';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import Image from 'next/image';
import { UsersIcon, TargetIcon, LightbulbIcon } from 'lucide-react';

export default function AboutPage() {
	const currentYear = new Date().getFullYear();

	return (
		<div className="flex flex-col min-h-screen bg-background text-foreground">
			{/* Header */}
			<header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
				<div className="container flex h-16 items-center max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<Link href="/" className="flex items-center gap-2 mr-6">
						<MailThreaderLogo className="h-7 w-7 text-primary" />
						<span className="text-xl font-semibold">
							Mail-Threader
						</span>
					</Link>
					<nav className="flex items-center space-x-2 sm:space-x-4 text-sm font-medium ml-auto">
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
					</nav>
				</div>
			</header>

			{/* Main Content */}
			<main className="flex-1 py-12 md:py-16 lg:py-20">
				<div className="container max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
					<Card className="shadow-xl overflow-hidden">
						<CardHeader className="text-center p-8 md:p-12 bg-muted/30">
							<div className="inline-flex items-center justify-center p-4 bg-primary/10 rounded-full mb-6 mx-auto">
								<MailThreaderLogo className="h-16 w-16 text-primary" />
							</div>
							<CardTitle className="text-4xl md:text-5xl font-bold tracking-tight">
								About Mail-Threader
							</CardTitle>
							<CardDescription className="mt-4 text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
								Unlocking Actionable Insights from Complex Email
								Archives with Intelligent Analysis.
							</CardDescription>
						</CardHeader>
						<CardContent className="p-8 md:p-12 space-y-10">
							<section className="space-y-4">
								<h2 className="text-2xl md:text-3xl font-semibold text-primary flex items-center gap-3">
									<TargetIcon className="h-8 w-8" />
									Our Mission
								</h2>
								<p className="text-lg text-foreground/80 leading-relaxed">
									At Mail-Threader, our mission is to empower
									researchers, analysts, legal professionals,
									and organizations to navigate and understand
									vast email datasets with unprecedented
									clarity and efficiency. We believe that
									hidden within complex communication archives
									are crucial insights, patterns, and
									narratives. Our AI-driven platform is
									designed to surface this valuable
									information, transforming raw data into
									actionable knowledge.
								</p>
							</section>

							<section className="space-y-4">
								<h2 className="text-2xl md:text-3xl font-semibold text-primary flex items-center gap-3">
									<LightbulbIcon className="h-8 w-8" />
									What We Do
								</h2>
								<p className="text-lg text-foreground/80 leading-relaxed">
									Mail-Threader provides a suite of powerful
									tools built on cutting-edge artificial
									intelligence and data processing techniques.
									We specialize in:
								</p>
								<ul className="list-disc list-inside space-y-2 text-lg text-foreground/80 leading-relaxed pl-4">
									<li>
										<strong>
											Intelligent Email Threading:
										</strong>{' '}
										Automatically grouping related emails
										into coherent conversational threads.
									</li>
									<li>
										<strong>
											AI-Powered Summarization:
										</strong>{' '}
										Condensing lengthy email chains and
										documents into concise, easy-to-digest
										summaries.
									</li>
									<li>
										<strong>
											Insightful Visualizations:
										</strong>{' '}
										Revealing communication patterns, topic
										clusters, and sentiment trends through
										interactive charts and graphs.
									</li>
									<li>
										<strong>
											Advanced Data Processing:
										</strong>{' '}
										Offering robust capabilities for
										cleaning, filtering, and structuring
										large email datasets for effective
										analysis.
									</li>
								</ul>
								<div className="grid md:grid-cols-2 gap-6 mt-6">
									<div className="rounded-lg overflow-hidden shadow-md">
										<Image
											src="https://picsum.photos/600/400?random=10"
											alt="Team working on data analysis"
											width={600}
											height={400}
											className="w-full h-auto object-cover"
											data-ai-hint="team collaboration"
										/>
									</div>
									<div className="rounded-lg overflow-hidden shadow-md">
										<Image
											src="https://picsum.photos/600/400?random=11"
											alt="Abstract representation of data threads"
											width={600}
											height={400}
											className="w-full h-auto object-cover"
											data-ai-hint="data network"
										/>
									</div>
								</div>
							</section>

							<section className="space-y-4">
								<h2 className="text-2xl md:text-3xl font-semibold text-primary flex items-center gap-3">
									<UsersIcon className="h-8 w-8" />
									Who We Serve
								</h2>
								<p className="text-lg text-foreground/80 leading-relaxed">
									Mail-Threader is designed for anyone who
									needs to make sense of large volumes of
									email data. This includes:
								</p>
								<ul className="list-disc list-inside space-y-2 text-lg text-foreground/80 leading-relaxed pl-4">
									<li>
										<strong>Academic Researchers:</strong>{' '}
										Studying communication patterns,
										historical events, or organizational
										behavior.
									</li>
									<li>
										<strong>Legal Professionals:</strong>{' '}
										Conducting e-discovery, investigations,
										and case preparation.
									</li>
									<li>
										<strong>Business Analysts:</strong>{' '}
										Identifying trends, risks, and
										opportunities within corporate
										communications.
									</li>
									<li>
										<strong>
											Journalists and Investigators:
										</strong>{' '}
										Uncovering stories and evidence from
										extensive email archives.
									</li>
									<li>
										<strong>
											Archivists and Historians:
										</strong>{' '}
										Preserving and making accessible digital
										communication records.
									</li>
								</ul>
							</section>
							<section className="text-center pt-6">
								<Button size="lg" asChild>
									<Link href="/contact">
										Contact Us To Learn More
									</Link>
								</Button>
							</section>
						</CardContent>
					</Card>
				</div>
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
