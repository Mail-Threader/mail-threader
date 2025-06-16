'use server';

import { db } from '@/db';
import { fileStorageTable } from '@/db/schema';
import { supabase } from '@/lib/supabaseClient';
import { revalidatePath } from 'next/cache';
import { eq } from 'drizzle-orm';

export async function uploadFileAction(
    userId: string,
    file: File,
    metadata: {
        name: string;
        size: string;
        type: string;
        lastModified: number;
    }
) {
    try {
        const fileId = crypto.randomUUID();
        const userFolder = userId.replace(/[^a-zA-Z0-9]/g, '_');
        const filePath = `${userFolder}/${Date.now()}_${file.name}`;

        // Upload to Supabase Storage
        const { data: uploadData, error: uploadError } = await supabase.storage
            .from('input-data')
            .upload(filePath, file, {
                cacheControl: '3600',
                upsert: false,
            });

        if (uploadError) {
            throw uploadError;
        }

        // Get the public URL
        const { data: { publicUrl } } = supabase.storage
            .from('input-data')
            .getPublicUrl(filePath);

        // Save file metadata to database
        await db.insert(fileStorageTable).values({
            id: fileId,
            name: metadata.name,
            userId,
            fileName: metadata.name,
            filePath,
            fileSize: metadata.size,
            fileType: metadata.type,
            fileUrl: publicUrl,
            fileThumbnail: '', // Add thumbnail generation if needed
            fileThumbnailUrl: '', // Add thumbnail URL if needed
            fileThumbnailPath: '', // Add thumbnail path if needed
            fileMetadata: {
                originalName: metadata.name,
                size: metadata.size,
                type: metadata.type,
                lastModified: metadata.lastModified,
            },
        });

        revalidatePath('/dashboard/upload-data');
        return { success: true, fileId };
    } catch (error) {
        console.error('Error uploading file:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Upload failed'
        };
    }
}

export async function fetchUserUploadsAction(userId: string) {
    try {
        const files = await db.query.fileStorageTable.findMany({
            where: eq(fileStorageTable.userId, userId),
            orderBy: (files, { desc }) => [desc(files.createdAt)],
        });

        return {
            success: true,
            files: files.map(file => ({
                id: file.id,
                name: file.name,
                uploadDate: file.createdAt?.toISOString() ?? new Date().toISOString(),
                size: file.fileSize,
                status: 'completed' as const,
            })),
        };
    } catch (error) {
        console.error('Error fetching user uploads:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Failed to fetch uploads',
        };
    }
}

export async function setActiveDatasetAction(userId: string, fileId: string) {
    try {
        // Verify the file belongs to the user
        const file = await db.query.fileStorageTable.findFirst({
            where: eq(fileStorageTable.id, fileId),
        });

        if (!file || file.userId !== userId) {
            return {
                success: false,
                error: 'File not found or access denied',
            };
        }

        // Here you would typically update a user preference or active dataset setting
        // For now, we'll just return success
        revalidatePath('/dashboard/upload-data');
        return { success: true };
    } catch (error) {
        console.error('Error setting active dataset:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Failed to set active dataset',
        };
    }
}
