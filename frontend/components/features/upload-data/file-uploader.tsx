'use client';

import { useCallback, useState } from 'react';
import { useDropzone, FileRejection } from 'react-dropzone';
import { useFileUploadStore } from '@/store/file-upload-store';
import { useAuthStore } from '@/store/auth-store';
import { uploadFileAction } from '@/actions/upload';
import { useToast } from '@/hooks/use-toast';
import { Progress } from '@/components/ui/progress';

// Define acceptable file types
const ACCEPTED_FILE_TYPES = {
	'text/*': ['.txt', '.csv', '.json'],
	'application/json': ['.json'],
	'application/vnd.ms-excel': ['.xls'],
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
};

const MAX_FILE_SIZE_MB = 50; // Define max file size in MB

export function FileUploader() {
	const { user } = useAuthStore();
	const { addUpload, updateUploadStatus } = useFileUploadStore();
	const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
	const { toast } = useToast();

	const onDrop = useCallback(async (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
		if (!user?.id) {
			toast({
				title: 'Error',
				description: 'You must be logged in to upload files',
				variant: 'destructive',
			});
			return;
		}

		for (const file of acceptedFiles) {
			const fileId = crypto.randomUUID();
			addUpload({
				id: fileId,
				name: file.name,
				uploadDate: new Date().toISOString(),
				size: formatFileSize(file.size),
				status: 'uploading',
			});

			try {
				// Simulate upload progress
				const progressInterval = setInterval(() => {
					setUploadProgress(prev => {
						const current = prev[fileId] || 0;
						if (current >= 90) {
							clearInterval(progressInterval);
							return prev;
						}
						return { ...prev, [fileId]: current + 10 };
					});
				}, 500);

				const result = await uploadFileAction(user.id, file, {
					name: file.name,
					size: formatFileSize(file.size),
					type: file.type,
					lastModified: file.lastModified,
				});

				clearInterval(progressInterval);
				setUploadProgress(prev => ({ ...prev, [fileId]: 100 }));

				if (result.success) {
					updateUploadStatus(fileId, 'completed');
					toast({
						title: 'Success',
						description: `${file.name} uploaded successfully`,
					});
				} else {
					updateUploadStatus(fileId, 'error', result.error);
					toast({
						title: 'Error',
						description: `Failed to upload ${file.name}: ${result.error}`,
						variant: 'destructive',
					});
				}
			} catch (error) {
				updateUploadStatus(fileId, 'error', error instanceof Error ? error.message : 'Upload failed');
				toast({
					title: 'Error',
					description: `Failed to upload ${file.name}`,
					variant: 'destructive',
				});
			}
		}

		// Handle rejected files
		for (const { file, errors } of rejectedFiles) {
			toast({
				title: 'Error',
				description: `Failed to upload ${file.name}: ${errors[0]?.message || 'Invalid file type'}`,
				variant: 'destructive',
			});
		}
	}, [user?.id, addUpload, updateUploadStatus, toast]);

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: ACCEPTED_FILE_TYPES,
		maxSize: MAX_FILE_SIZE_MB * 1024 * 1024,
	});

	return (
		<div className="space-y-4">
			<div
				{...getRootProps()}
				className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'
					}`}
			>
				<input {...getInputProps()} />
				{isDragActive ? (
					<p className="text-primary">Drop the files here...</p>
				) : (
					<div className="space-y-2">
						<p className="text-gray-600">Drag and drop files here, or click to select files</p>
						<p className="text-sm text-gray-500">Supported formats: .txt, .csv, .json, .xls, .xlsx</p>
					</div>
				)}
			</div>

			{Object.entries(uploadProgress).map(([fileId, progress]) => (
				<div key={fileId} className="space-y-2">
					<div className="flex justify-between text-sm">
						<span>Uploading...</span>
						<span>{progress}%</span>
					</div>
					<Progress value={progress} className="h-2" />
				</div>
			))}
		</div>
	);
}

function formatFileSize(bytes: number): string {
	if (bytes === 0) return '0 Bytes';
	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
