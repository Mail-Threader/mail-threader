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
import { useFileUploadStore } from '@/store/file-upload-store';
import { useAuthStore } from '@/store/auth-store';
import { parseISO, isWithinInterval, isValid } from 'date-fns';
import { useMemo, useEffect } from 'react';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';

export default function UploadDataPage() {
	const { keywords, dateRange } = useFilterStore();
	const { user } = useAuthStore();
	const { uploads, activeDataset, isLoading, error, fetchUserUploads, setActiveDataset } = useFileUploadStore();

	useEffect(() => {
		if (user?.id) {
			fetchUserUploads(user.id);
		}
	}, [user?.id, fetchUserUploads]);

	const filteredUploads = useMemo(() => {
		let data = [...uploads];

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
	}, [uploads, keywords, dateRange]);

	const handleSetActiveDataset = async (fileId: string) => {
		if (!user?.id) {
			toast.error('You must be logged in to set an active dataset');
			return;
		}

		await setActiveDataset(user.id, fileId);
		toast.success('Active dataset updated successfully');
	};

	if (!user) {
		return (
			<div className="flex items-center justify-center min-h-screen">
				<p className="text-lg">Please log in to access this page.</p>
			</div>
		);
	}

	return (
		<div className="container mx-auto px-4 py-8">
			<h1 className="text-3xl font-bold mb-8">Upload Data</h1>

			<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
				<div>
					<Card className="p-6">
						<h2 className="text-xl font-semibold mb-4">Upload New File</h2>
						<FileUploader />
					</Card>
				</div>

				<div>
					<Card className="p-6">
						<h2 className="text-xl font-semibold mb-4">Your Uploads</h2>
						{isLoading ? (
							<p>Loading...</p>
						) : error ? (
							<p className="text-red-500">{error}</p>
						) : uploads.length === 0 ? (
							<p>No uploads yet. Upload your first file to get started!</p>
						) : (
							<div className="space-y-4">
								{uploads.map((upload) => (
									<div
										key={upload.id}
										className="flex items-center justify-between p-4 border rounded-lg"
									>
										<div>
											<p className="font-medium">{upload.name}</p>
											<p className="text-sm text-gray-500">
												{upload.size} • {new Date(upload.uploadDate).toLocaleDateString()}
											</p>
										</div>
										<Button
											variant={activeDataset === upload.id ? 'default' : 'outline'}
											onClick={() => handleSetActiveDataset(upload.id)}
											disabled={isLoading}
										>
											{activeDataset === upload.id ? 'Active Dataset' : 'Set as Active'}
										</Button>
									</div>
								))}
							</div>
						)}
					</Card>
				</div>
			</div>
		</div>
	);
}
