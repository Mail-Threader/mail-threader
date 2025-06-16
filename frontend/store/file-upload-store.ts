import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { fetchUserUploadsAction, setActiveDatasetAction } from '@/actions/upload';

export interface FileUpload {
    id: string;
    name: string;
    uploadDate: string;
    size: string;
    status: 'uploading' | 'completed' | 'error';
    error?: string;
}

interface FileUploadState {
    uploads: FileUpload[];
    activeDataset: string | null;
    isLoading: boolean;
    error: string | null;
    addUpload: (upload: FileUpload) => void;
    updateUploadStatus: (id: string, status: FileUpload['status'], error?: string) => void;
    fetchUserUploads: (userId: string) => Promise<void>;
    setActiveDataset: (userId: string, fileId: string) => Promise<void>;
    clearUploads: () => void;
}

export const useFileUploadStore = create<FileUploadState>()(
    devtools((set, get) => ({
        uploads: [],
        activeDataset: null,
        isLoading: false,
        error: null,

        addUpload: (upload) => {
            set((state) => ({
                uploads: [...state.uploads, upload],
            }));
        },

        updateUploadStatus: (id, status, error) => {
            set((state) => ({
                uploads: state.uploads.map((upload) =>
                    upload.id === id ? { ...upload, status, error } : upload
                ),
            }));
        },

        fetchUserUploads: async (userId) => {
            set({ isLoading: true, error: null });
            try {
                const result = await fetchUserUploadsAction(userId);

                if (result.success) {
                    set({ uploads: result.files, isLoading: false });
                } else {
                    set({ error: result.error ?? 'Failed to fetch uploads', isLoading: false });
                }
            } catch (error) {
                set({
                    error: error instanceof Error ? error.message : 'Failed to fetch uploads',
                    isLoading: false,
                });
            }
        },

        setActiveDataset: async (userId, fileId) => {
            set({ isLoading: true, error: null });
            try {
                const result = await setActiveDatasetAction(userId, fileId);
                if (result.success) {
                    set({ activeDataset: fileId, isLoading: false });
                } else {
                    set({ error: result.error ?? 'Failed to set active dataset', isLoading: false });
                }
            } catch (error) {
                set({
                    error: error instanceof Error ? error.message : 'Failed to set active dataset',
                    isLoading: false,
                });
            }
        },

        clearUploads: () => {
            set({ uploads: [], activeDataset: null, error: null });
        },
    }))
);
