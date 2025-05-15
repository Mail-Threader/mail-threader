'use client';

import { useState, type DragEvent, type ChangeEvent, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { UploadCloudIcon, FileIcon, XIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';

// Define acceptable file types
const ACCEPTED_FILE_TYPES = {
	'text/csv': ['.csv'],
	'message/rfc822': ['.eml'],
	'application/mbox': ['.mbox'],
	'application/octet-stream': ['.pkl'], // Common fallback for .pkl, extension check is primary
	'application/vnd.ms-outlook': ['.pst'],
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': [
		'.xlsx',
	],
	'application/zip': ['.zip'],
};
const MAX_FILE_SIZE_MB = 100; // Example: 100MB limit

// Generate the string for the <input accept> attribute
const acceptAttributeValue = Object.entries(ACCEPTED_FILE_TYPES)
	.map(([mimeType, extensions]) => extensions.join(',')) // Primarily use extensions for accept
	.join(',');

// Generate the display string for supported types
const supportedTypesDisplay = Object.values(ACCEPTED_FILE_TYPES)
	.flat()
	.map((ext) => ext.substring(1).toUpperCase())
	.sort()
	.join(', ');

export function FileUploader() {
	const [isDragging, setIsDragging] = useState(false);
	const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
	const { toast } = useToast();
	const router = useRouter();

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
			e.target.value = ''; // Reset file input to allow re-uploading the same file
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

		toast({
			title: 'Upload Started',
			description: `Uploading ${selectedFiles.length} file(s)... (This is a demo)`,
		});

		await new Promise((resolve) => setTimeout(resolve, 2000));

		toast({
			title: 'Upload Successful',
			description: `${selectedFiles.length} file(s) "uploaded". Redirecting to data view...`,
		});
		setSelectedFiles([]);
		router.push('/dashboard/data-view');
	}, [selectedFiles, toast, router]);

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
				onClick={() => document.getElementById('fileInput')?.click()}
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
									onClick={() => removeFile(index)}
									aria-label="Remove file"
								>
									<XIcon className="w-4 h-4" />
								</Button>
							</li>
						))}
					</ul>
					<div className="flex justify-end pt-2">
						<Button
							onClick={handleUpload}
							disabled={selectedFiles.length === 0}
						>
							Upload {selectedFiles.length} File(s)
						</Button>
					</div>
				</div>
			)}
		</div>
	);
}
