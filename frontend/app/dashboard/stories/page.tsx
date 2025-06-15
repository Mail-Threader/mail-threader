'use client';

import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from '@/components/ui/accordion';
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LightbulbIcon, Share2Icon, ThumbsUpIcon } from 'lucide-react';
import Image from 'next/image';
import { useFilterStore } from '@/store/filter-store';
import { parseISO, isWithinInterval, isValid } from 'date-fns';
import { useMemo } from 'react';

interface Story {
	id: string;
	title: string;
	narrative: string;
	keyInsights: string[];
	relatedVisualizations: string[]; // IDs or links to visualizations
	dateDiscovered: string; // Expecting YYYY-MM-DD format
	tags: string[];
}

const placeholderStoriesData: Story[] = [
	{
		id: 'STY001',
		title: 'The Rise and Fall of LJM: A Tale of Off-Balance-Sheet Deception',
		narrative:
			"This story uncovers the intricate web of Special Purpose Entities (SPEs), particularly LJM, created by Andrew Fastow. Emails reveal how these entities were used to hide debt, inflate earnings, and enrich executives, ultimately contributing to Enron's collapse.",
		keyInsights: [
			'Misleading financial reporting through SPEs.',
			'Conflicts of interest for executives involved in LJM.',
			'Lack of transparency and oversight.',
		],
		relatedVisualizations: [
			'Email Volume by Sender (Fastow)',
			'Financial Irregularities Timeline',
		],
		dateDiscovered: '2001-10-25',
		tags: ['LJM', 'SPE', 'Fastow', 'Deception', 'Finance'],
	},
	{
		id: 'STY002',
		title: "California's Power Play: Enron's Role in the Energy Crisis",
		narrative:
			"Emails suggest Enron traders exploited California's deregulated energy market, contributing to rolling blackouts and price spikes. This story explores the strategies discussed internally, including 'Death Star' and 'Get Shorty', and the subsequent public and regulatory fallout.",
		keyInsights: [
			'Market manipulation tactics.',
			'Impact on California consumers.',
			'Regulatory loopholes exploited.',
		],
		relatedVisualizations: [
			'Energy Price Spike Chart',
			'Trader Communication Network',
		],
		dateDiscovered: '2001-04-10',
		tags: [
			'California',
			'Energy Crisis',
			'Market Manipulation',
			'Regulation',
		],
	},
	{
		id: 'STY003',
		title: 'The Culture of Secrecy: How Internal Communication Masked Problems',
		narrative:
			'An analysis of communication patterns reveals a culture where bad news was suppressed and overly optimistic projections were favored. This story highlights how a lack of candid internal discussion allowed critical issues to fester.',
		keyInsights: [
			'Suppression of negative information.',
			'Overemphasis on positive spin.',
			'Breakdown of internal controls.',
		],
		relatedVisualizations: [
			'Sentiment Analysis Over Time',
			'Keyword Frequency (Optimism vs. Concern)',
		],
		dateDiscovered: '2001-11-15',
		tags: [
			'Corporate Culture',
			'Communication',
			'Secrecy',
			'Internal Controls',
		],
	},
];

export default function StoryExplorerPage() {
	const { keywords, dateRange } = useFilterStore();

	const filteredStories = useMemo(() => {
		let data = [...placeholderStoriesData];

		if (keywords) {
			const lowercasedKeywords = keywords.toLowerCase();
			data = data.filter(
				(story) =>
					story.title.toLowerCase().includes(lowercasedKeywords) ||
					story.narrative
						.toLowerCase()
						.includes(lowercasedKeywords) ||
					story.tags.some((tag) =>
						tag.toLowerCase().includes(lowercasedKeywords),
					) ||
					story.keyInsights.some((insight) =>
						insight.toLowerCase().includes(lowercasedKeywords),
					),
			);
		}

		if (dateRange?.from || dateRange?.to) {
			data = data.filter((story) => {
				const storyDate = parseISO(story.dateDiscovered);
				if (!isValid(storyDate)) return false;

				const fromDate = dateRange.from;
				const toDate = dateRange.to
					? new Date(dateRange.to.setHours(23, 59, 59, 999))
					: undefined;

				if (fromDate && toDate) {
					return isWithinInterval(storyDate, {
						start: fromDate,
						end: toDate,
					});
				}
				if (fromDate) {
					return storyDate >= fromDate;
				}
				if (toDate) {
					return storyDate <= toDate;
				}
				return true;
			});
		}
		return data;
	}, [placeholderStoriesData, keywords, dateRange]);

	return (
		<div className="space-y-6">
			<Card>
				<CardHeader>
					<CardTitle>Generated Stories & Insights</CardTitle>
					<CardDescription>
						Discover compelling narratives and key insights derived
						from the Enron email dataset.
						{(keywords || dateRange?.from) &&
							` Filtering by: ${
								keywords ? `keywords "${keywords}"` : ''
							}${keywords && dateRange?.from ? ' and ' : ''}${
								dateRange?.from ? `dates` : ''
							}.`}
					</CardDescription>
				</CardHeader>
				<CardContent>
					{filteredStories.length > 0 ? (
						<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
							{filteredStories.map((story) => (
								<Card key={story.id} className="flex flex-col">
									<CardHeader>
										<div className="mb-2">
											<Image
												src={`https://placehold.co/400x200.png`}
												alt={story.title}
												width={400}
												height={200}
												className="rounded-md w-full h-auto object-cover"
												data-ai-hint={`${
													story.tags[0]
												} ${
													story.tags[1] || 'abstract'
												}`}
											/>
										</div>
										<CardTitle className="text-lg">
											{story.title}
										</CardTitle>
										<CardDescription>
											Discovered: {story.dateDiscovered}
										</CardDescription>
									</CardHeader>
									<CardContent className="flex-grow">
										<p className="text-sm text-muted-foreground mb-3 line-clamp-3">
											{story.narrative}
										</p>
										<Accordion
											type="single"
											collapsible
											className="w-full text-sm"
										>
											<AccordionItem value="insights">
												<AccordionTrigger className="py-2 text-primary hover:no-underline">
													<div className="flex items-center gap-2">
														<LightbulbIcon className="h-4 w-4" />{' '}
														Key Insights
													</div>
												</AccordionTrigger>
												<AccordionContent className="pt-2 pl-6">
													<ul className="list-disc space-y-1 text-muted-foreground">
														{story.keyInsights.map(
															(insight, idx) => (
																<li key={idx}>
																	{insight}
																</li>
															),
														)}
													</ul>
												</AccordionContent>
											</AccordionItem>
										</Accordion>
									</CardContent>
									<CardFooter className="flex justify-between items-center pt-4 border-t">
										<div className="flex gap-1 flex-wrap">
											{story.tags
												.slice(0, 3)
												.map((tag) => (
													<Badge
														variant="outline"
														key={tag}
														className="mb-1"
													>
														{tag}
													</Badge>
												))}
										</div>
										<div className="flex gap-2">
											<Button
												variant="ghost"
												size="icon"
												aria-label="Like story"
											>
												<ThumbsUpIcon className="h-4 w-4" />
											</Button>
											<Button
												variant="ghost"
												size="icon"
												aria-label="Share story"
											>
												<Share2Icon className="h-4 w-4" />
											</Button>
										</div>
									</CardFooter>
								</Card>
							))}
						</div>
					) : (
						<p className="text-muted-foreground text-center py-8">
							No stories match your current filters.
						</p>
					)}
				</CardContent>
			</Card>
		</div>
	);
}
