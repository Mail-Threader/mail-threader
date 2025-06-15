'use client';

import { type ColumnDef } from '@tanstack/react-table';
import { type SummarizedEmail } from '@/db/schema';
import { type DataTableRowAction } from '@/types/data-table';
import { format } from 'date-fns';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Eye } from 'lucide-react';

interface GetSummarizedEmailTableColumnsProps {
    setRowAction: (action: DataTableRowAction<SummarizedEmail> | null) => void;
    onViewDetails: (email: SummarizedEmail) => void;
}

export function getSummarizedEmailTableColumns({
    setRowAction,
    onViewDetails,
}: GetSummarizedEmailTableColumnsProps): ColumnDef<SummarizedEmail>[] {
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
            accessorKey: 'sentiment',
            header: 'Sentiment',
            enableColumnFilter: true,
            meta: {
                variant: 'select',
                label: 'Sentiment',
                placeholder: 'Filter by sentiment...',
                options: [
                    { label: 'Positive', value: 'POSITIVE' },
                    { label: 'Neutral', value: 'NEUTRAL' },
                    { label: 'Negative', value: 'NEGATIVE' },
                ],
            },
            cell: ({ row }) => {
                const sentiment = row.original.sentiment;
                return (
                    <div className="flex max-w-[100px] flex-col gap-1">
                        <Badge
                            variant={
                                sentiment === 'POSITIVE'
                                    ? 'default'
                                    : sentiment === 'NEGATIVE'
                                        ? 'destructive'
                                        : 'secondary'
                            }
                        >
                            {sentiment || 'NEUTRAL'}
                        </Badge>
                    </div>
                );
            },
        },
        {
            accessorKey: 'cluster',
            header: 'Cluster',
            enableColumnFilter: true,
            meta: {
                variant: 'text',
                label: 'Cluster',
                placeholder: 'Filter by cluster...',
            },
            cell: ({ row }) => {
                return (
                    <div className="flex max-w-[100px] flex-col gap-1">
                        <Badge variant="outline">
                            {row.original.cluster || 'N/A'}
                        </Badge>
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
                        size="icon"
                        onClick={() => onViewDetails(row.original)}
                    >
                        <Eye className="h-4 w-4" />
                    </Button>
                );
            },
        },
    ];
}
