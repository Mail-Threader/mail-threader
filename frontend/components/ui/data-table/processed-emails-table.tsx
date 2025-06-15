'use client';

import { useState, useMemo, useEffect } from 'react';
import type { ProcessedEmail } from '@/db/schema';
import type { DataTableRowAction } from '@/types/data-table';
import { DataTable } from './data-table';
import { DataTableToolbar } from './data-table-toolbar';
import { DataTableFilterMenu } from './data-table-filter-menu';
import { DataTableSortList } from './data-table-sort-list';
import { ProcessedEmailDetailsDialog } from './processed-email-details-dialog';
import { getProcessedEmailTableColumns } from './processed-emails-table-columns';
import { useDataTable } from '@/hooks/use-data-table';
import { getProcessedEmailsAction } from '@/actions/data';

interface ProcessedEmailsTableProps {
    initialData: { data: ProcessedEmail[]; total: number };
}

export function Table({ initialData }: ProcessedEmailsTableProps) {
    const [selectedEmail, setSelectedEmail] = useState<ProcessedEmail | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [rowAction, setRowAction] = useState<DataTableRowAction<ProcessedEmail> | null>(null);
    const [data, setData] = useState<ProcessedEmail[]>(initialData.data);
    const [total, setTotal] = useState<number>(initialData.total);
    const [loading, setLoading] = useState(false);

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
        pageCount: Math.ceil(total / (data.length > 0 ? data.length : 10)),
        enableAdvancedFilter: false,
        initialState: {
            sorting: [{ id: 'date', desc: true }],
            columnPinning: { right: ['actions'] },
            pagination: {
                pageSize: initialData.data.length || 10,
                pageIndex: 0,
            },
        },
        getRowId: (originalRow) => {
            const messageId = originalRow.messageId ?? '';
            const originalMessageId = originalRow.originalMessageId ?? '';
            const mainId = originalRow.mainId ?? '';
            const date = originalRow.date ?? '';
            return `${messageId}-${originalMessageId}-${mainId}-${date}`;
        },
        shallow: false,
        clearOnDefault: true,
    });

    // Fetch data when pagination changes
    useEffect(() => {
        const { pageIndex, pageSize } = table.getState().pagination;
        setLoading(true);
        getProcessedEmailsAction({
            page: pageIndex + 1,
            perPage: pageSize,
            sort: [{ id: 'date', desc: true }],
            filters: [],
            joinOperator: 'and',
            filterFlag: 'advancedFilters',
            subject: '',
        }).then((result) => {
            setData(result.data);
            setTotal(result.total);
            setLoading(false);
        });
    }, [table.getState().pagination.pageIndex, table.getState().pagination.pageSize]);

    return (
        <>
            {loading && <div className="text-center py-4">Loading...</div>}
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
