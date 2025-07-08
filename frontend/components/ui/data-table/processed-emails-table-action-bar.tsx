'use client';

import { type ProcessedData } from '@/db/schema';
import type { Table } from '@tanstack/react-table';

import {
	DataTableActionBar,
	DataTableActionBarSelection,
} from './data-table-action-bar';

interface ProcessedEmailTableActionBarProps {
	table: Table<ProcessedData>;
}

export function ProcessedEmailTableActionBar({
	table,
}: ProcessedEmailTableActionBarProps) {
	const rows = table.getFilteredSelectedRowModel().rows;

	return (
		<DataTableActionBar table={table} visible={rows.length > 0}>
			<DataTableActionBarSelection table={table} />
		</DataTableActionBar>
	);
}
