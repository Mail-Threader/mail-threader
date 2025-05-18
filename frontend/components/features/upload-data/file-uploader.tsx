'use client';

import { useState, type DragEvent, type ChangeEvent, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { UploadCloudIcon, FileIcon, XIcon, Loader2Icon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabaseClient';
import { useAuthStore } from '@/store/auth-store';

// Define acceptable file types
const ACCEPTED_FILE_TYPES = {
	'text/csv': ['.csv'],
	'application/octet-stream': ['.pkl'],
	'application/vnd.ms-outlook': ['.pst'],
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': [
		'.xlsx',
	],
	'application/zip': ['.zip'],
};
const MAX_FILE_SIZE_MB = 100;
const SUPABASE_BUCKET_NAME = 'dev-data'; // Define your bucket name

const acceptAttributeValue = Object.values(ACCEPTED_FILE_TYPES)
	.flat()
	.join(',');

const supportedTypesDisplay = Object.values(ACCEPTED_FILE_TYPES)
	.flat()
	.map((ext) => ext.substring(1).toUpperCase())
	.sort()
	.join(', ');

export function FileUploader() {
	const [isDragging, setIsDragging] = useState(false);
	const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
	const [isUploading, setIsUploading] = useState(false);
	const { toast } = useToast();
	const router = useRouter();
	const { user } = useAuthStore();

	const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(true);
	};

	const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(false);
	};

	const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		e.stopPropagation();
		if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
			setIsDragging(true);
		}
	};

	const validateFile = (file: File): string | null => {
		if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
			return `File "${file.name}" is too large. Max size is ${MAX_FILE_SIZE_MB}MB.`;
		}

		const fileName = file.name.toLowerCase();
		let extension = fileName.split('.').pop();

		if (!extension) {
			return `File "${file.name}" has no extension and is not supported.`;
		}
		extension = `.${extension}`;

		const allAcceptedExtensions = Object.values(ACCEPTED_FILE_TYPES).flat();

		if (!allAcceptedExtensions.includes(extension)) {
			return `File type for "${
				file.name
			}" (extension ${extension}) is not supported. Supported types: ${allAcceptedExtensions.join(
				', ',
			)}.`;
		}
		return null;
	};

	const processFiles = (files: FileList | null) => {
		if (files) {
			const newFiles: File[] = [];
			const errors: string[] = [];
			Array.from(files).forEach((file) => {
				const error = validateFile(file);
				if (error) {
					errors.push(error);
				} else {
					newFiles.push(file);
				}
			});

			if (errors.length > 0) {
				toast({
					title: 'File Validation Error',
					description: errors.join('\n'),
					variant: 'destructive',
				});
			}

			setSelectedFiles((prevFiles) => [...prevFiles, ...newFiles]);
		}
	};

	const handleDrop = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		e.stopPropagation();
		setIsDragging(false);
		processFiles(e.dataTransfer.files);
	};

	const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
		processFiles(e.target.files);
		if (e.target) {
			e.target.value = '';
		}
	};

	const removeFile = (index: number) => {
		setSelectedFiles((prevFiles) =>
			prevFiles.filter((_, i) => i !== index),
		);
	};

	const handleUpload = useCallback(async () => {
		if (selectedFiles.length === 0) {
			toast({
				title: 'No files selected',
				description: 'Please select files to upload.',
				variant: 'destructive',
			});
			return;
		}

		setIsUploading(true);
		toast({
			title: 'Upload In Progress',
			description: `Uploading ${selectedFiles.length} file(s) to Supabase...`,
		});

		const uploadPromises = selectedFiles.map(async (file) => {
			// Sanitize email to use as a folder name, or use a user ID if available
			// For simplicity, using email here. Replace with a proper user ID in production.
			const userFolder = user?.id
				? user.id.replace(/[^a-zA-Z0-9]/g, '_')
				: 'anonymous';
			const filePath = `${userFolder}/${Date.now()}_${file.name}`; // Add timestamp to avoid overwrites

			// const {} = await supabase.storage

			// check if the bucket exists with getBucket
			const { data: bucketData, error: bucketError } =
				await supabase.storage.getBucket(SUPABASE_BUCKET_NAME);

			console.log({
				bucketData,
				bucketError,
			});

			if (bucketError) {
				console.error('Error fetching bucket:', bucketError);
				toast({
					title: 'Bucket Error',
					description: 'Could not access the storage bucket.',
					variant: 'destructive',
				});
				return;
			}

			if (!bucketData) {
				console.error('Bucket not found:', SUPABASE_BUCKET_NAME);
				toast({
					title: 'Bucket Not Found',
					description: 'The specified storage bucket does not exist.',
					variant: 'destructive',
				});

				// create the bucket if it doesn't exist
				const { error: createBucketError } =
					await supabase.storage.createBucket(SUPABASE_BUCKET_NAME, {
						public: false,
					});

				if (createBucketError) {
					console.error('Error creating bucket:', createBucketError);
					toast({
						title: 'Bucket Creation Error',
						description: 'Could not create the storage bucket.',
						variant: 'destructive',
					});
					return;
				}
			}

			const { data, error } = await supabase.storage
				.from(SUPABASE_BUCKET_NAME)
				.upload(filePath, file, {
					cacheControl: '3600',
					upsert: false, // Set to true if you want to overwrite files with the same name
				});

			if (error) {
				console.error(`Error uploading ${file.name}:`, error);
				return {
					success: false,
					name: file.name,
					error: error.message,
				};
			}
			console.log(`Successfully uploaded ${file.name}:`, data);
			return { success: true, name: file.name };
		});

		try {
			const results = await Promise.all(uploadPromises);

			const successfulUploads = results.filter((r) => r && r.success);
			const failedUploads = results.filter((r) => r && !r.success);

			if (failedUploads.length > 0) {
				toast({
					title: 'Upload Partially Failed',
					description: `Failed to upload ${
						failedUploads.length
					} file(s): ${failedUploads
						.map((f) => f && f.name)
						.join(', ')}.`,
					variant: 'destructive',
				});
			}

			if (successfulUploads.length > 0) {
				toast({
					title: 'Upload Successful',
					description: `${
						successfulUploads.length
					} file(s) uploaded to Supabase. ${
						failedUploads.length > 0 ? 'Some files failed.' : ''
					}`,
				});
			}

			setSelectedFiles([]); // Clear selection after upload attempt
			// Potentially redirect or update UI, e.g., refresh the list of past uploads if it were dynamic
			// router.push('/dashboard/data-view'); // Kept original redirect for now
		} catch (error) {
			console.error('General upload error:', error);
			toast({
				title: 'Upload Error',
				description: 'An unexpected error occurred during upload.',
				variant: 'destructive',
			});
		} finally {
			setIsUploading(false);
		}
	}, [selectedFiles, toast, router, user]);

	return (
		<div className="space-y-6">
			<div
				className={cn(
					'flex flex-col items-center justify-center w-full p-8 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary/70 transition-colors',
					isDragging
						? 'border-primary bg-primary/10'
						: 'border-border bg-card',
				)}
				onDragEnter={handleDragEnter}
				onDragLeave={handleDragLeave}
				onDragOver={handleDragOver}
				onDrop={handleDrop}
				onClick={() =>
					!isUploading &&
					document.getElementById('fileInput')?.click()
				}
			>
				<UploadCloudIcon
					className={cn(
						'w-16 h-16 mb-4',
						isDragging ? 'text-primary' : 'text-muted-foreground',
					)}
				/>
				<p className="mb-2 text-sm text-muted-foreground">
					<span className="font-semibold">Click to upload</span> or
					drag and drop
				</p>
				<p className="text-xs text-muted-foreground">
					Supported: {supportedTypesDisplay}. Max {MAX_FILE_SIZE_MB}MB
					per file.
				</p>
				<Input
					id="fileInput"
					type="file"
					multiple
					className="hidden"
					onChange={handleFileChange}
					accept={acceptAttributeValue}
					disabled={isUploading}
				/>
			</div>

			{selectedFiles.length > 0 && (
				<div className="space-y-3">
					<h3 className="text-lg font-medium">Selected Files:</h3>
					<ul className="space-y-2">
						{selectedFiles.map((file, index) => (
							<li
								key={index}
								className="flex items-center justify-between p-3 border rounded-md bg-muted/50"
							>
								<div className="flex items-center gap-3">
									<FileIcon className="w-5 h-5 text-muted-foreground" />
									<span className="text-sm font-medium truncate max-w-xs sm:max-w-md md:max-w-lg">
										{file.name}
									</span>
									<span className="text-xs text-muted-foreground">
										({(file.size / 1024 / 1024).toFixed(2)}{' '}
										MB)
									</span>
								</div>
								<Button
									variant="ghost"
									size="icon"
									onClick={() =>
										!isUploading && removeFile(index)
									}
									aria-label="Remove file"
									disabled={isUploading}
								>
									<XIcon className="w-4 h-4" />
								</Button>
							</li>
						))}
					</ul>
					<div className="flex justify-end pt-2">
						<Button
							onClick={handleUpload}
							disabled={selectedFiles.length === 0 || isUploading}
						>
							{isUploading ? (
								<>
									<Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
									Uploading...
								</>
							) : (
								`Upload ${selectedFiles.length} File(s)`
							)}
						</Button>
					</div>
				</div>
			)}
		</div>
	);
}
