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

interface EmailData {
	id: string;
	from: string;
	to: string;
	subject: string;
	date: string;
	snippet: string;
	status: 'Processed' | 'Pending' | 'Error';
}

const placeholderData: EmailData[] = [
	{
		id: 'EM001',
		from: 'jeff.skilling@enron.com',
		to: 'john.doe@example.com',
		subject: 'Q4 Projections',
		date: '2000-10-15',
		snippet: 'Please review the attached Q4 financial projections...',
		status: 'Processed',
	},
	{
		id: 'EM002',
		from: 'andrew.fastow@enron.com',
		to: 'jane.smith@example.com',
		subject: 'Project Raptor Update',
		date: '2000-11-03',
		snippet:
			'The Project Raptor initiative is moving forward as planned...',
		status: 'Processed',
	},
	{
		id: 'EM003',
		from: 'kenneth.lay@enron.com',
		to: 'board.members@enron.com',
		subject: 'Strategic Review Meeting',
		date: '2000-12-01',
		snippet:
			'Reminder: Strategic review meeting scheduled for next week...',
		status: 'Pending',
	},
	{
		id: 'EM004',
		from: 'lou.pai@enron.com',
		to: 'mark.rich@example.com',
		subject: 'Asset Sale Inquiry',
		date: '2001-01-20',
		snippet: 'Following up on our conversation regarding asset sales...',
		status: 'Processed',
	},
	{
		id: 'EM005',
		from: 'rebecca.mark@enron.com',
		to: 'international.team@enron.com',
		subject: 'International Expansion Plans',
		date: '2001-02-10',
		snippet: "Let's discuss the new international expansion proposals.",
		status: 'Error',
	},
];

export default function DataViewPage() {
	return (
		<div className="space-y-6">
			<Card>
				<CardHeader>
					<CardTitle>Processed Email Data</CardTitle>
					<CardDescription>
						View and explore the data processed during the
						preparation stage of the Enron email dataset.
					</CardDescription>
				</CardHeader>
				<CardContent>
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
							{placeholderData.map((email) => (
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
													: email.status === 'Pending'
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
							src="https://picsum.photos/600/400?random=1"
							alt="Data Distribution Chart"
							width={600}
							height={400}
							className="w-full h-auto object-cover"
							data-ai-hint="data distribution"
						/>
					</div>
					<div className="rounded-lg overflow-hidden shadow-lg">
						<Image
							src="https://picsum.photos/600/400?random=2"
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
