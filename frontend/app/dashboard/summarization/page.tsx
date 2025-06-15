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
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { FileTextIcon, ZapIcon } from 'lucide-react';
import Image from 'next/image';
import { useFilterStore } from '@/store/filter-store';
import { parseISO, isWithinInterval, isValid } from 'date-fns';
import { useMemo } from 'react';

interface SummaryItem {
	id: string;
	title: string;
	content: string;
	keywords: string[];
	dateGenerated: string; // Expecting YYYY-MM-DD format
	sourceEmails: number;
}

const placeholderSummariesData: SummaryItem[] = [
	{
		id: 'SUM001',
		title: 'Q4 Financial Performance Overview',
		content:
			'The fourth quarter showed strong revenue growth driven by new energy contracts. However, operating expenses also increased, impacting overall profitability. Key concerns revolve around derivative accounting practices.',
		keywords: ['finance', 'revenue', 'Q4', 'profitability'],
		dateGenerated: '2001-01-20',
		sourceEmails: 152,
	},
	{
		id: 'SUM002',
		title: 'Project Raptor: Risk Assessment',
		content:
			'Project Raptor involves complex special purpose entities (SPEs) designed to hedge risk and manage debt. Analysis indicates potential accounting irregularities and off-balance-sheet liabilities that could pose significant financial risk.',
		keywords: ['Project Raptor', 'SPEs', 'risk', 'accounting'],
		dateGenerated: '2001-03-05',
		sourceEmails: 88,
	},
	{
		id: 'SUM003',
		title: 'California Energy Crisis Communications',
		content:
			"Internal communications reveal discussions about strategies to manage public perception and regulatory scrutiny during the California energy crisis. Topics include price manipulation allegations and Enron's role in market dynamics.",
		keywords: ['California', 'energy crisis', 'regulation', 'PR'],
		dateGenerated: '2001-05-15',
		sourceEmails: 230,
	},
];

export default function SummarizationPage() {
	const { keywords, dateRange } = useFilterStore();

	const filteredSummaries = useMemo(() => {
		let data = [...placeholderSummariesData];

		if (keywords) {
			const lowercasedKeywords = keywords.toLowerCase();
			data = data.filter(
				(summary) =>
					summary.title.toLowerCase().includes(lowercasedKeywords) ||
					summary.content
						.toLowerCase()
						.includes(lowercasedKeywords) ||
					summary.keywords.some((kw) =>
						kw.toLowerCase().includes(lowercasedKeywords),
					),
			);
		}

		if (dateRange?.from || dateRange?.to) {
			data = data.filter((summary) => {
				const summaryDate = parseISO(summary.dateGenerated);
				if (!isValid(summaryDate)) return false;

				const fromDate = dateRange.from;
				const toDate = dateRange.to
					? new Date(dateRange.to.setHours(23, 59, 59, 999))
					: undefined;

				if (fromDate && toDate) {
					return isWithinInterval(summaryDate, {
						start: fromDate,
						end: toDate,
					});
				}
				if (fromDate) {
					return summaryDate >= fromDate;
				}
				if (toDate) {
					return summaryDate <= toDate;
				}
				return true;
			});
		}
		return data;
	}, [placeholderSummariesData, keywords, dateRange]);

	return (
		<div className="space-y-6">
			{/* <Card>
				<CardHeader>
					<CardTitle>Email Summaries</CardTitle>
					<CardDescription>
						Explore AI-generated summaries focusing on pertinent
						details from the Enron email dataset.
						{(keywords || dateRange?.from) &&
							` Filtering by: ${
								keywords ? `keywords "${keywords}"` : ''
							}${keywords && dateRange?.from ? ' and ' : ''}${
								dateRange?.from ? `dates` : ''
							}.`}
					</CardDescription>
				</CardHeader>
				<CardContent>
					{filteredSummaries.length > 0 ? (
						<Accordion type="single" collapsible className="w-full">
							{filteredSummaries.map((summary, index) => (
								<AccordionItem
									value={`item-${index}`}
									key={summary.id}
								>
									<AccordionTrigger>
										<div className="flex items-center gap-3">
											<FileTextIcon className="h-5 w-5 text-primary" />
											<span className="font-medium">
												{summary.title}
											</span>
										</div>
									</AccordionTrigger>
									<AccordionContent className="space-y-3 pl-8">
										<p className="text-muted-foreground">
											{summary.content}
										</p>
										<div className="text-xs text-muted-foreground space-y-1">
											<p>
												<strong>Date Generated:</strong>{' '}
												{summary.dateGenerated}
											</p>
											<p>
												<strong>
													Source Emails Analyzed:
												</strong>{' '}
												{summary.sourceEmails}
											</p>
											<div className="flex items-center gap-2 pt-1">
												<strong>Keywords:</strong>
												{summary.keywords.map(
													(keyword) => (
														<Badge
															variant="secondary"
															key={keyword}
														>
															{keyword}
														</Badge>
													),
												)}
											</div>
										</div>
									</AccordionContent>
								</AccordionItem>
							))}
						</Accordion>
					) : (
						<p className="text-muted-foreground text-center py-8">
							No summaries match your current filters.
						</p>
					)}
				</CardContent>
			</Card> */}
		</div>
	);
}
