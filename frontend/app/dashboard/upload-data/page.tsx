'use client';

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { FileUploader } from '@/components/features/upload-data/file-uploader';
import { Label } from '@/components/ui/label';
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
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
import { useMemo } from 'react';

interface PastUpload {
	id: string;
	name: string;
	uploadDate: string; // Expecting YYYY-MM-DD format
	size: string;
}

const placeholderPastUploadsData: PastUpload[] = [
	{
		id: 'ds001',
		name: 'Enron Full Archive (mbox)',
		uploadDate: '2024-07-15',
		size: '5.2 GB',
	},
	{
		id: 'ds002',
		name: 'Q3 Financial Emails (csv)',
		uploadDate: '2024-07-20',
		size: '15 MB',
	},
	{
		id: 'ds003',
		name: 'Project Alpha Comms (zip)',
		uploadDate: '2024-07-22',
		size: '120 MB',
	},
	{
		id: 'ds004',
		name: 'Executive Correspondence (eml)',
		uploadDate: '2024-07-25',
		size: '250 MB',
	},
];

export default function UploadDataPage() {
	const { keywords, dateRange } = useFilterStore();

	const filteredPastUploads = useMemo(() => {
		let data = [...placeholderPastUploadsData];

		if (keywords) {
			const lowercasedKeywords = keywords.toLowerCase();
			data = data.filter((upload) =>
				upload.name.toLowerCase().includes(lowercasedKeywords),
			);
		}

		if (dateRange?.from || dateRange?.to) {
			data = data.filter((upload) => {
				const uploadDateObj = parseISO(upload.uploadDate);
				if (!isValid(uploadDateObj)) return false;

				const fromDate = dateRange.from;
				const toDate = dateRange.to
					? new Date(dateRange.to.setHours(23, 59, 59, 999))
					: undefined;

				if (fromDate && toDate) {
					return isWithinInterval(uploadDateObj, {
						start: fromDate,
						end: toDate,
					});
				}
				if (fromDate) {
					return uploadDateObj >= fromDate;
				}
				if (toDate) {
					return uploadDateObj <= toDate;
				}
				return true;
			});
		}
		return data;
	}, [placeholderPastUploadsData, keywords, dateRange]);

	return (
		<div className="space-y-6">
			<Card>
				<CardHeader>
					<CardTitle>Upload Your Email Data</CardTitle>
					<CardDescription>
						Drag and drop your email data files (e.g., .mbox, .pst,
						.eml folders) or click to select files.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<FileUploader />
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Processing Queue</CardTitle>
					<CardDescription>
						View the status of your uploaded files.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<p className="text-sm text-muted-foreground">
						No files currently processing. Uploaded files will
						appear here with their status.
					</p>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Manage and Use Uploaded Datasets</CardTitle>
					<CardDescription>
						Select an active dataset for analysis or review your
						upload history.
						{(keywords || dateRange?.from) &&
							` Filtering history by: ${
								keywords ? `keywords "${keywords}"` : ''
							}${keywords && dateRange?.from ? ' and ' : ''}${
								dateRange?.from ? `dates` : ''
							}.`}
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-6">
					<div className="space-y-2">
						<Label
							htmlFor="active-dataset-select"
							className="text-base font-medium"
						>
							Active Dataset for Dashboard
						</Label>
						<div className="flex flex-col sm:flex-row sm:items-center gap-2">
							<Select>
								<SelectTrigger
									id="active-dataset-select"
									className="sm:flex-grow"
								>
									<SelectValue placeholder="Choose a dataset to make active" />
								</SelectTrigger>
								<SelectContent>
									{placeholderPastUploadsData.map(
										(upload) => (
											<SelectItem
												key={upload.id}
												value={upload.id}
											>
												{upload.name} (Uploaded:{' '}
												{upload.uploadDate})
											</SelectItem>
										),
									)}
									{placeholderPastUploadsData.length ===
										0 && (
										<SelectItem value="no-data" disabled>
											No past uploads available
										</SelectItem>
									)}
								</SelectContent>
							</Select>
							<Button
								className="w-full sm:w-auto flex-shrink-0"
								disabled={
									placeholderPastUploadsData.length === 0
								}
							>
								Load Selected Dataset
							</Button>
						</div>
					</div>

					<div>
						<h3 className="text-lg font-semibold mb-3">
							Upload History
						</h3>
						<div className="rounded-md border">
							<Table>
								<TableHeader>
									<TableRow>
										<TableHead>Dataset Name</TableHead>
										<TableHead>Upload Date</TableHead>
										<TableHead>Size</TableHead>
										<TableHead className="text-right">
											Actions
										</TableHead>
									</TableRow>
								</TableHeader>
								<TableBody>
									{filteredPastUploads.length > 0 ? (
										filteredPastUploads.map((upload) => (
											<TableRow key={upload.id}>
												<TableCell className="font-medium">
													{upload.name}
												</TableCell>
												<TableCell>
													{upload.uploadDate}
												</TableCell>
												<TableCell>
													{upload.size}
												</TableCell>
												<TableCell className="text-right">
													<Button
														variant="outline"
														size="sm"
													>
														Details
													</Button>
												</TableCell>
											</TableRow>
										))
									) : (
										<TableRow>
											<TableCell
												colSpan={4}
												className="h-24 text-center text-muted-foreground"
											>
												No past uploads found matching
												your filters.
											</TableCell>
										</TableRow>
									)}
								</TableBody>
							</Table>
						</div>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
