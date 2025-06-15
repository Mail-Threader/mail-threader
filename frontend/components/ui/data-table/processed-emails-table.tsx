'use client';

import { type ProcessedEmail } from '@/db/schema';
import { useState, useMemo } from 'react';
import { DataTable } from './data-table';
import { getProcessedEmailTableColumns } from './processed-emails-table-columns';
import { ProcessedEmailDetailsDialog } from './processed-email-details-dialog';
import type { DataTableRowAction } from '@/types/data-table';
import { useDataTable } from '@/hooks/use-data-table';
import { use } from 'react';
import type { getProcessedEmails } from '@/lib/queries';
import { DataTableToolbar } from './data-table-toolbar';
import { DataTableFilterMenu } from './data-table-filter-menu';
import { DataTableSortList } from './data-table-sort-list';

interface ProcessedEmailsTableProps {
    promises: Promise<[Awaited<ReturnType<typeof getProcessedEmails>>]>;
}

export function Table({ promises }: ProcessedEmailsTableProps) {
    const [selectedEmail, setSelectedEmail] = useState<ProcessedEmail | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [rowAction, setRowAction] = useState<DataTableRowAction<ProcessedEmail> | null>(null);

    const [{ data, pageCount }] = use(promises);

    const columns = useMemo(
        () =>
            getProcessedEmailTableColumns({
                setRowAction,
                onViewDetails: (email) => {
                    setSelectedEmail(email);
                    setDialogOpen(true);
                },
            }),
        [setRowAction]
    );

    const { table, shallow, debounceMs, throttleMs } = useDataTable({
        data,
        columns,
        pageCount,
        enableAdvancedFilter: false,
        initialState: {
            sorting: [{ id: 'date', desc: true }],
            columnPinning: { right: ['actions'] },
        },
        getRowId: (originalRow) => originalRow.messageId ?? '',
        shallow: false,
        clearOnDefault: true,
    });

    return (
        <>
            <DataTable table={table}>
                <DataTableToolbar table={table}>
                    <DataTableFilterMenu
                        table={table}
                        shallow={shallow}
                        debounceMs={debounceMs}
                        throttleMs={throttleMs}
                    />
                    <DataTableSortList table={table} align="end" />
                </DataTableToolbar>
            </DataTable>
            <ProcessedEmailDetailsDialog
                email={selectedEmail}
                open={dialogOpen}
                onOpenChange={setDialogOpen}
            />
        </>
    );
}
