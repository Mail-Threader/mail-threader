import { Table as ProcessedEmailsTable } from '@/components/ui/data-table/processed-emails-table';
import { getProcessedEmails } from '@/lib/queries';

export default function DataViewPage() {
	return (
		<main className="px-8 py-4">
			<h3 className="text-3xl">Cleaned & processed Data</h3>
			<p className="text-muted-foreground mb-4 text-lg">
				View and explore the data processed during the preparation stage
				of the email dataset.
			</p>
			<ProcessedEmailsTable
				promises={Promise.all([
					getProcessedEmails({
						page: 1,
						perPage: 10,
						sort: [],
						filters: [],
						joinOperator: 'and',
						filterFlag: 'advancedFilters',
						subject: '',
					})
				])}
			/>
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
		</main>
	);
}
