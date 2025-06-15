'use client';

import { create } from 'zustand';
import type { DateRange } from 'react-day-picker';

interface FilterState {
	keywords: string;
	dateRange: DateRange | undefined;
	setKeywords: (keywords: string) => void;
	setDateRange: (dateRange: DateRange | undefined) => void;
	resetFilters: () => void;
}

const initialDateRange: DateRange | undefined = undefined;

export const useFilterStore = create<FilterState>((set) => ({
	keywords: '',
	dateRange: initialDateRange,
	setKeywords: (keywords) => set({ keywords }),
	setDateRange: (dateRange) => set({ dateRange }),
	resetFilters: () => set({ keywords: '', dateRange: initialDateRange }),
}));
