'use client';

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import Image from 'next/image';
import { useFilterStore } from '@/store/filter-store';
import { parseISO, isWithinInterval, isValid } from 'date-fns';
import { useMemo } from 'react';

interface EmailData {
	id: string;
	from: string;
	to: string;
	subject: string;
	date: string; // Expecting YYYY-MM-DD format
	snippet: string;
	status: 'Processed' | 'Pending' | 'Error';
}

const placeholderEmailData: EmailData[] = [
	{
		id: 'EM001',
		from: 'jeff.skilling@enron.com',
		to: 'john.doe@example.com',
		subject: 'Q4 Projections',
		date: '2025-05-15',
		snippet: 'Please review the attached Q4 financial projections...',
		status: 'Processed',
	},
	{
		id: 'EM002',
		from: 'andrew.fastow@enron.com',
		to: 'jane.smith@example.com',
		subject: 'Project Raptor Update',
		date: '2025-05-03',
		snippet:
			'The Project Raptor initiative is moving forward as planned...',
		status: 'Processed',
	},
	{
		id: 'EM003',
		from: 'kenneth.lay@enron.com',
		to: 'board.members@enron.com',
		subject: 'Strategic Review Meeting',
		date: '2025-05-01',
		snippet:
			'Reminder: Strategic review meeting scheduled for next week...',
		status: 'Pending',
	},
	{
		id: 'EM004',
		from: 'lou.pai@enron.com',
		to: 'mark.rich@example.com',
		subject: 'Asset Sale Inquiry',
		date: '2025-05-20',
		snippet: 'Following up on our conversation regarding asset sales...',
		status: 'Processed',
	},
	{
		id: 'EM005',
		from: 'rebecca.mark@enron.com',
		to: 'international.team@enron.com',
		subject: 'International Expansion Plans',
		date: '2025-05-10',
		snippet: "Let's discuss the new international expansion proposals.",
		status: 'Error',
	},
];

export default function DataViewPage() {
	const { keywords, dateRange } = useFilterStore();

	const filteredData = useMemo(() => {
		let data = [...placeholderEmailData];

		if (keywords) {
			const lowercasedKeywords = keywords.toLowerCase();
			data = data.filter(
				(email) =>
					email.subject.toLowerCase().includes(lowercasedKeywords) ||
					email.from.toLowerCase().includes(lowercasedKeywords) ||
					email.to.toLowerCase().includes(lowercasedKeywords) ||
					email.snippet.toLowerCase().includes(lowercasedKeywords),
			);
		}

		if (dateRange?.from || dateRange?.to) {
			data = data.filter((email) => {
				const emailDate = parseISO(email.date);
				if (!isValid(emailDate)) return false;

				const fromDate = dateRange.from;
				const toDate = dateRange.to
					? new Date(dateRange.to.setHours(23, 59, 59, 999))
					: undefined;

				if (fromDate && toDate) {
					return isWithinInterval(emailDate, {
						start: fromDate,
						end: toDate,
					});
				}
				if (fromDate) {
					return emailDate >= fromDate;
				}
				if (toDate) {
					return emailDate <= toDate;
				}
				return true;
			});
		}
		return data;
	}, [placeholderEmailData, keywords, dateRange]);

	return (
		<div className="space-y-6">
			<Card>
				<CardHeader>
					<CardTitle>Processed Email Data</CardTitle>
					<CardDescription>
						View and explore the data processed during the
						preparation stage of the Enron email dataset.
						{(keywords || dateRange?.from) &&
							` Filtering by: ${
								keywords ? `keywords "${keywords}"` : ''
							}${keywords && dateRange?.from ? ' and ' : ''}${
								dateRange?.from ? `dates` : ''
							}.`}
					</CardDescription>
				</CardHeader>
				<CardContent>
					{filteredData.length > 0 ? (
						<Table>
							<TableHeader>
								<TableRow>
									<TableHead>Email ID</TableHead>
									<TableHead>From</TableHead>
									<TableHead>To</TableHead>
									<TableHead>Subject</TableHead>
									<TableHead>Date</TableHead>
									<TableHead>Status</TableHead>
									<TableHead className="text-right">
										Snippet
									</TableHead>
								</TableRow>
							</TableHeader>
							<TableBody>
								{filteredData.map((email) => (
									<TableRow key={email.id}>
										<TableCell className="font-medium">
											{email.id}
										</TableCell>
										<TableCell>{email.from}</TableCell>
										<TableCell>{email.to}</TableCell>
										<TableCell>{email.subject}</TableCell>
										<TableCell>{email.date}</TableCell>
										<TableCell>
											<Badge
												variant={
													email.status === 'Processed'
														? 'default'
														: email.status ===
														  'Pending'
														? 'secondary'
														: 'destructive'
												}
											>
												{email.status}
											</Badge>
										</TableCell>
										<TableCell className="text-right text-muted-foreground truncate max-w-xs">
											{email.snippet}
										</TableCell>
									</TableRow>
								))}
							</TableBody>
						</Table>
					) : (
						<p className="text-muted-foreground text-center py-8">
							No data matches your current filters.
						</p>
					)}
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Data Overview</CardTitle>
					<CardDescription>
						Visual representation of data characteristics.
					</CardDescription>
				</CardHeader>
				<CardContent className="grid md:grid-cols-2 gap-6">
					<div className="rounded-lg overflow-hidden shadow-lg">
						<Image
							src="https://placehold.co/600x400.png"
							alt="Data Distribution Chart"
							width={600}
							height={400}
							className="w-full h-auto object-cover"
							data-ai-hint="data distribution"
						/>
					</div>
					<div className="rounded-lg overflow-hidden shadow-lg">
						<Image
							src="https://placehold.co/600x400.png"
							alt="Data Timeline Chart"
							width={600}
							height={400}
							className="w-full h-auto object-cover"
							data-ai-hint="data timeline"
						/>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
