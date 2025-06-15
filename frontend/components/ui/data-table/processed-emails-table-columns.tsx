'use client';

import type { ProcessedEmail } from '@/db/schema';
import type { DataTableRowAction } from '@/types/data-table';
import { ColumnDef } from '@tanstack/react-table';
import { format, isValid, parse } from 'date-fns';
import { MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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

function parseRobustDate(date: string): Date | null {
    if (!date) return null;
    // Try ISO, then MM/DD/YY, then fallback
    let parsed = new Date(date);
    if (isValid(parsed)) return parsed;
    parsed = parse(date, 'MM/dd/yy', new Date());
    if (isValid(parsed)) return parsed;
    parsed = parse(date, 'M/d/yy', new Date());
    if (isValid(parsed)) return parsed;
    parsed = parse(date, 'yyyy-MM-dd', new Date());
    if (isValid(parsed)) return parsed;
    return null;
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
                    <div className="flex max-w-[300px] flex-col gap-1">
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
            accessorKey: 'type',
            header: () => <div className="text-center">Type</div>,
            enableColumnFilter: true,
            meta: {
                variant: 'text',
                label: 'Type',
                placeholder: 'Filter by type...',
            },
            cell: ({ row }) => {
                const type = row.original.type;
                const getTypeVariant = (type: string | null | undefined) => {
                    if (!type) return 'secondary';
                    const lowerType = type.toLowerCase();
                    if (lowerType === 'main') return 'default';
                    if (lowerType === 'forwarded') return 'secondary';
                    if (lowerType === 'original') return 'outline';
                    return 'secondary';
                };

                const capitalizeType = (type: string | null | undefined) => {
                    if (!type) return 'N/A';
                    return type.charAt(0).toUpperCase() + type.slice(1).toLowerCase();
                };

                return (
                    <div className="flex max-w-[150px] flex-col gap-1 items-center justify-center">
                        <Badge
                            variant={getTypeVariant(type)}
                            className={`font-medium ${type?.toLowerCase() === 'main'
                                ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                : type?.toLowerCase() === 'forwarded'
                                    ? 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                                    : type?.toLowerCase() === 'original'
                                        ? 'bg-green-100 text-green-700 hover:bg-green-200'
                                        : ''
                                }`}
                        >
                            {capitalizeType(type)}
                        </Badge>
                    </div>
                );
            },
        },
        {
            accessorKey: 'date',
            header: () => <div className="text-right min-w-[140px]">Date</div>,
            enableColumnFilter: true,
            meta: {
                variant: 'date',
                label: 'Date',
                placeholder: 'Filter by date...',
            },
            cell: ({ row }) => {
                const date = row.original.date;
                if (!date) return <span className="truncate text-right block min-w-[140px]">—</span>;
                const parsed = parseRobustDate(date);
                if (!parsed) return <span className="truncate text-right block min-w-[140px]">—</span>;
                return <span className="truncate text-right block min-w-[140px]">{format(parsed, 'PPP')}</span>;
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
