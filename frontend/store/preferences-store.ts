'use client';

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Theme = 'theme-light' | 'dark' | 'system';

interface PreferencesState {
	theme: Theme;
	setTheme: (theme: Theme) => void;
}

export const usePreferencesStore = create<PreferencesState>()(
	persist(
		(set) => ({
			theme: 'system' as Theme, // Default theme
			setTheme: (theme: Theme) => set({ theme }),
		}),
		{
			name: 'app-preferences-storage', // Name of the item in localStorage
			storage: createJSONStorage(() => localStorage),
		},
	),
);
