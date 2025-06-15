'use client';

import { useFilterStore } from '@/store/filter-store';
import { parseISO, isWithinInterval, isValid } from 'date-fns';
import { useMemo } from 'react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { StoryMetrics } from '@/components/features/stories/StoryMetrics';
import { StoryTimeline } from '@/components/features/stories/StoryTimeline';
import { StoryNetwork } from '@/components/features/stories/StoryNetwork';
import { StoryTopics } from '@/components/features/stories/StoryTopics';
import { StoryEmails } from '@/components/features/stories/StoryEmails';
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';


interface Story {
	id: string;
	title: string;
	narrative: string;
	keyInsights: string[];
	relatedVisualizations: string[]; // IDs or links to visualizations
	dateDiscovered: string; // Expecting YYYY-MM-DD format
	tags: string[];
	metrics?: {
		influenceScore?: number;
		avgResponseTime?: number;
		emailCount?: number;
		peakActivity?: {
			day: string;
			count: number;
		};
	};
	timeline?: {
		startDate: string;
		endDate: string;
		events: Array<{
			date: string;
			description: string;
			significance: number;
		}>;
	};
	network?: {
		nodes: Array<{
			id: string;
			label: string;
			value: number;
		}>;
		links: Array<{
			source: string;
			target: string;
			value: number;
		}>;
	};
	topics?: Array<{
		name: string;
		count: number;
		sentiment: number;
		keywords: string[];
	}>;
	relatedEmails?: Array<{
		subject: string;
		date: string;
		from: string;
		to: string;
		bodyPreview: string;
		category: string;
	}>;
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
		metrics: {
			influenceScore: 0.85,
			avgResponseTime: 4.2,
			emailCount: 1245,
			peakActivity: {
				day: 'Wednesday',
				count: 45,
			},
		},
		timeline: {
			startDate: '2000-01-01',
			endDate: '2001-12-31',
			events: [
				{
					date: '2000-03-15',
					description: 'LJM2 partnership formation',
					significance: 0.9,
				},
				{
					date: '2000-06-20',
					description: 'First suspicious transaction',
					significance: 0.7,
				},
				{
					date: '2001-08-15',
					description: 'Internal audit raises concerns',
					significance: 0.8,
				},
			],
		},
		network: {
			nodes: [
				{ id: 'fastow', label: 'Andrew Fastow', value: 0.9 },
				{ id: 'skilling', label: 'Jeff Skilling', value: 0.8 },
				{ id: 'lay', label: 'Ken Lay', value: 0.7 },
			],
			links: [
				{ source: 'fastow', target: 'skilling', value: 0.8 },
				{ source: 'fastow', target: 'lay', value: 0.6 },
				{ source: 'skilling', target: 'lay', value: 0.7 },
			],
		},
		topics: [
			{
				name: 'Financial Engineering',
				count: 450,
				sentiment: 0.3,
				keywords: ['SPE', 'partnership', 'off-balance-sheet', 'debt'],
			},
			{
				name: 'Risk Management',
				count: 320,
				sentiment: -0.2,
				keywords: ['risk', 'exposure', 'hedge', 'derivatives'],
			},
		],
		relatedEmails: [
			{
				subject: 'LJM2 Partnership Structure',
				date: '2000-03-15',
				from: 'andrew.fastow@enron.com',
				to: 'jeff.skilling@enron.com',
				bodyPreview: 'Proposed structure for the LJM2 partnership...',
				category: 'Partnership Formation',
			},
			{
				subject: 'Re: LJM2 Partnership Structure',
				date: '2000-03-16',
				from: 'jeff.skilling@enron.com',
				to: 'andrew.fastow@enron.com',
				bodyPreview: 'Approved. Let\'s proceed with the structure...',
				category: 'Partnership Formation',
			},
		],
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
					story.narrative.toLowerCase().includes(lowercasedKeywords) ||
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
		<div className="px-4 md:px-16 lg:px-32 py-6 w-full">
			<h2 className="text-2xl font-bold mt-4 mb-2 text-foreground">Generated Stories & Insights</h2>
			<p className="text-muted-foreground mb-4">
				Discover compelling narratives and key insights derived from the Enron email dataset.
				{(keywords || dateRange?.from) &&
					` Filtering by: ${keywords ? `keywords "${keywords}"` : ''
					}${keywords && dateRange?.from ? ' and ' : ''}${dateRange?.from ? `dates` : ''
					}.`}
			</p>
			<TooltipProvider>
				<Accordion type="multiple" className="w-full">
					{filteredStories.length > 0 ? (
						filteredStories.map((story) => (
							<AccordionItem key={story.id} value={story.id}>
								<AccordionTrigger className="flex flex-row items-center justify-between gap-4 py-4 px-4 bg-muted hover:bg-accent rounded-xl border border-border mb-2 shadow-sm transition-all group">
									<div className="flex flex-col md:flex-row md:items-center gap-2 w-full">
										<span className="font-semibold text-lg text-left group-hover:text-blue-700 transition-colors text-foreground">{story.title}</span>
										<span className="text-xs text-muted-foreground ml-2">{story.dateDiscovered}</span>
										<span className="flex gap-1 flex-wrap ml-2">
											{story.tags.slice(0, 3).map((tag) => (
												<Badge variant="outline" key={tag} className="mb-1">
													{tag}
												</Badge>
											))}
										</span>
										{story.metrics?.influenceScore !== undefined && (
											<span className="ml-2 text-xs text-blue-700 font-bold flex items-center gap-1">
												<Tooltip>
													<TooltipTrigger asChild>
														<span className="underline decoration-dotted cursor-help flex items-center">
															Influence: {story.metrics.influenceScore.toFixed(2)}
															<Info className="h-3 w-3 ml-1" />
														</span>
													</TooltipTrigger>
													<TooltipContent className="max-w-xs text-xs">
														Influence Score is a composite metric (degree, betweenness, PageRank) indicating this actor's importance in the Enron email network. Higher values mean more central or influential roles in communication.
													</TooltipContent>
												</Tooltip>
											</span>
										)}
									</div>
								</AccordionTrigger>
								<AccordionContent className="bg-card border border-border rounded-xl p-6 mb-4 shadow-md text-foreground">
									<div className="mb-4">
										<div className="text-base leading-relaxed mb-2 text-foreground">{story.narrative}</div>
										{story.metrics && <StoryMetrics metrics={story.metrics} />}
									</div>
									{story.timeline && <StoryTimeline timeline={story.timeline} />}
									{story.network && <div className="mt-6"><StoryNetwork network={story.network} /></div>}
									{story.topics && <div className="mt-6"><StoryTopics topics={story.topics} /></div>}
									{story.relatedEmails && <div className="mt-6"><StoryEmails emails={story.relatedEmails} /></div>}
								</AccordionContent>
							</AccordionItem>
						))
					) : (
						<p className="text-muted-foreground text-center py-8">
							No stories match your current filters.
						</p>
					)}
				</Accordion>
			</TooltipProvider>
		</div>
	);
}
