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
import { useFilterStore } from '@/store/filter-store';
import { parseISO, isWithinInterval, isValid } from 'date-fns';
import { useEffect, useMemo, useState } from 'react';
import { ProcessedEmail } from '@/db/schema';
import { getProcessedEmails } from '@/actions/data';
// import { DataTable } from '@/components/ui/data-table';

export default function DataViewPage() {
	const [data, setData] = useState<ProcessedEmail[]>([]);
	const [pageNum, setPageNum] = useState(0);

	const { keywords, dateRange } = useFilterStore();

	const filteredData = useMemo(() => {
		let _data = [...data];

		if (keywords) {
			const lowercasedKeywords = keywords.toLowerCase();
			_data = _data.filter(
				(email) =>
					(email.subject &&
						email.subject
							.toLowerCase()
							.includes(lowercasedKeywords)) ||
					(email.from &&
						email.from
							.toLowerCase()
							.includes(lowercasedKeywords)) ||
					(email.to &&
						email.to.toLowerCase().includes(lowercasedKeywords)) ||
					(email.body &&
						email.body.toLowerCase().includes(lowercasedKeywords)),
			);
		}

		if (dateRange?.from || dateRange?.to) {
			_data = data.filter((email) => {
				if (!email.date) return false;
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
	}, [data, keywords, dateRange]);

	useEffect(() => {
		// Simulate fetching data from a database or API
		const fetchData = async () => {
			const response = await getProcessedEmails(pageNum);
			if (response.length > 0) {
				setData(response);
			} else {
				setPageNum(0);
			}
		};
		fetchData();
	}, []);

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
				{/* <CardContent>
					<DataTable data={filteredData} />
				</CardContent> */}
				<CardContent>
					{filteredData.length > 0 ? (
						<Table>
							<TableHeader>
								<TableRow>
									<TableHead>message_id</TableHead>
									<TableHead>original_message_id</TableHead>
									<TableHead>main_id</TableHead>
									<TableHead>filename</TableHead>
									<TableHead>type</TableHead>
									<TableHead>date</TableHead>
									<TableHead>from</TableHead>
									<TableHead>X-From</TableHead>
									<TableHead>X-To</TableHead>
									<TableHead>original_sender</TableHead>
									<TableHead>original_Date</TableHead>
									<TableHead>to</TableHead>
									<TableHead>subject</TableHead>
									<TableHead>cc</TableHead>
									<TableHead>X-cc</TableHead>
									<TableHead>body</TableHead>
								</TableRow>
							</TableHeader>
							<TableBody>
								{filteredData.map((email, i) => (
									<TableRow key={i}>
										<TableCell className="font-medium truncate max-w-3xs">
											{email.messageId}
										</TableCell>
										<TableCell className="font-medium truncate max-w-3xs">
											{email.originalMessageId}
										</TableCell>
										<TableCell className="font-medium truncate max-w-3xs">
											{email.mainId}
										</TableCell>
										<TableCell>{email.filename}</TableCell>
										<TableCell>{email.type}</TableCell>
										<TableCell>{email.date}</TableCell>
										<TableCell>{email.from}</TableCell>
										<TableCell>{email.xFrom}</TableCell>
										<TableCell className="text-right text-muted-foreground truncate max-w-xs">
											{email.xTo}
										</TableCell>
										<TableCell>
											{email.originalSender}
										</TableCell>
										<TableCell>
											{email.originalDate}
										</TableCell>
										<TableCell>{email.to}</TableCell>
										<TableCell>{email.subject}</TableCell>
										<TableCell>{email.cc}</TableCell>
										<TableCell className="text-right text-muted-foreground truncate max-w-xs">
											{email.xCc}
										</TableCell>
										{/* <TableCell>{email.body}</TableCell> */}
										<TableCell className="text-right text-muted-foreground truncate max-w-xs">
											{email.body}
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

			{/* <Card>
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
			</Card> */}
		</div>
	);
}
