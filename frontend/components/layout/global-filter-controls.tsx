'use client';

import * as React from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { DateRangePicker } from '@/components/ui/date-range-picker';
import { useFilterStore } from '@/store/filter-store';
import { XIcon, SearchIcon } from 'lucide-react';
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';

export function GlobalFilterControls() {
	const { keywords, dateRange, setKeywords, setDateRange, resetFilters } =
		useFilterStore();
	const [localKeywords, setLocalKeywords] = React.useState(keywords);

	// Effect to sync localKeywords when global keywords change (e.g. on reset)
	React.useEffect(() => {
		setLocalKeywords(keywords);
	}, [keywords]);

	const handleApplyFilters = () => {
		setKeywords(localKeywords);
		// dateRange is already live in the store via DateRangePicker's onDateChange
	};

	const handleResetFilters = () => {
		setLocalKeywords(''); // Reset local input
		resetFilters(); // Reset store
	};

	const canReset = !!localKeywords || !!dateRange?.from || !!dateRange?.to;

	return (
		<Card className="mb-6 shadow-md">
			<CardHeader>
				<CardTitle className="text-lg flex items-center">
					<SearchIcon className="mr-2 h-5 w-5" />
					Global Filters
				</CardTitle>
				<CardDescription>
					Apply filters across relevant dashboard pages. Keywords are
					applied on 'Search'. Dates apply instantly.
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-4">
				<div className="grid grid-cols-1 md:grid-cols-7 gap-4 items-end">
					<div className="md:col-span-3 space-y-1.5">
						<Label htmlFor="global-keywords">Keywords</Label>
						<Input
							id="global-keywords"
							placeholder="Search by keywords..."
							value={localKeywords}
							onChange={(e) => setLocalKeywords(e.target.value)}
							onKeyDown={(e) => {
								if (e.key === 'Enter') handleApplyFilters();
							}}
						/>
					</div>
					<div className="md:col-span-2 space-y-1.5">
						<Label htmlFor="global-date-range">Date Range</Label>
						<DateRangePicker
							date={dateRange}
							onDateChange={setDateRange}
						/>
					</div>
					<div className="md:col-span-2 flex items-end gap-2">
						<Button
							onClick={handleApplyFilters}
							className="w-full md:w-auto flex-grow"
						>
							<SearchIcon className="mr-2 h-4 w-4" />
							Search
						</Button>
						<Button
							variant="outline"
							onClick={handleResetFilters}
							disabled={!canReset}
							className="w-full md:w-auto flex-grow"
						>
							<XIcon className="mr-2 h-4 w-4" />
							Reset
						</Button>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
