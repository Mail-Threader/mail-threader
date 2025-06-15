'use client';

import type { ProcessedEmail } from '@/db/schema';
import type { DataTableRowAction } from '@/types/data-table';
import { ColumnDef } from '@tanstack/react-table';
import { format } from 'date-fns';
import { MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface GetProcessedEmailTableColumnsProps {
    setRowAction: (action: DataTableRowAction<ProcessedEmail> | null) => void;
    onViewDetails: (email: ProcessedEmail) => void;
}

export function getProcessedEmailTableColumns({
    setRowAction,
    onViewDetails,
}: GetProcessedEmailTableColumnsProps): ColumnDef<ProcessedEmail>[] {
    return [
        {
            accessorKey: 'subject',
            header: 'Subject',
            enableColumnFilter: true,
            meta: {
                variant: 'text',
                label: 'Subject',
                placeholder: 'Filter by subject...',
            },
            cell: ({ row }) => {
                return (
                    <div className="flex max-w-[500px] flex-col gap-1">
                        <span className="truncate font-medium">
                            {row.original.subject || 'No Subject'}
                        </span>
                    </div>
                );
            },
        },
        {
            accessorKey: 'from',
            header: 'From',
            enableColumnFilter: true,
            meta: {
                variant: 'text',
                label: 'From',
                placeholder: 'Filter by sender...',
            },
            cell: ({ row }) => {
                return (
                    <div className="flex max-w-[300px] flex-col gap-1">
                        <span className="truncate">{row.original.from}</span>
                    </div>
                );
            },
        },
        {
            accessorKey: 'to',
            header: 'To',
            enableColumnFilter: true,
            meta: {
                variant: 'text',
                label: 'To',
                placeholder: 'Filter by recipient...',
            },
            cell: ({ row }) => {
                return (
                    <div className="flex max-w-[300px] flex-col gap-1">
                        <span className="truncate">{row.original.to}</span>
                    </div>
                );
            },
        },
        {
            accessorKey: 'date',
            header: 'Date',
            enableColumnFilter: true,
            meta: {
                variant: 'date',
                label: 'Date',
                placeholder: 'Filter by date...',
            },
            cell: ({ row }) => {
                return (
                    <div className="flex max-w-[200px] flex-col gap-1">
                        <span className="truncate">
                            {row.original.date
                                ? format(new Date(row.original.date), 'PPP')
                                : 'N/A'}
                        </span>
                    </div>
                );
            },
        },
        {
            id: 'actions',
            cell: ({ row }) => {
                return (
                    <Button
                        variant="ghost"
                        className="flex data-[state=open]:bg-muted"
                        onClick={() => onViewDetails(row.original)}
                    >
                        View Details
                        <span className="sr-only">View Details</span>
                    </Button>
                );
            },
        },
    ];
}
