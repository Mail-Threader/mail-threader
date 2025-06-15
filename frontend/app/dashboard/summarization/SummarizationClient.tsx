'use client';

import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from '@/components/ui/accordion';
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { FileTextIcon } from 'lucide-react';
import { useFilterStore } from '@/store/filter-store';
import { useMemo, useState } from 'react';
import { DataTable } from '@/components/ui/data-table/data-table';
import { getSummarizedEmailTableColumns } from '@/components/ui/data-table/summarized-emails-table-columns';
import { SummarizedEmailDetailsDialog } from '@/components/ui/data-table/summarized-email-details-dialog';
import type { SummarizedEmail } from '@/db/schema';
import type { DataTableRowAction } from '@/types/data-table';
import { useDataTable } from '@/hooks/use-data-table';
import { DataTableToolbar } from '@/components/ui/data-table/data-table-toolbar';
import { DataTableFilterMenu } from '@/components/ui/data-table/data-table-filter-menu';
import { DataTableSortList } from '@/components/ui/data-table/data-table-sort-list';

interface SummaryItem {
    id: string;
    title: string;
    content: string;
    keywords: string[];
    dateGenerated: string;
    sourceEmails: number;
}

const placeholderSummariesData: SummaryItem[] = [
    {
        id: 'SUM001',
        title: 'Q4 Financial Performance Overview',
        content:
            'The fourth quarter showed strong revenue growth driven by new energy contracts. However, operating expenses also increased, impacting overall profitability. Key concerns revolve around derivative accounting practices.',
        keywords: ['finance', 'revenue', 'Q4', 'profitability'],
        dateGenerated: '2001-01-20',
        sourceEmails: 152,
    },
    {
        id: 'SUM002',
        title: 'Project Raptor: Risk Assessment',
        content:
            'Project Raptor involves complex special purpose entities (SPEs) designed to hedge risk and manage debt. Analysis indicates potential accounting irregularities and off-balance-sheet liabilities that could pose significant financial risk.',
        keywords: ['Project Raptor', 'SPEs', 'risk', 'accounting'],
        dateGenerated: '2001-03-05',
        sourceEmails: 88,
    },
    {
        id: 'SUM003',
        title: 'California Energy Crisis Communications',
        content:
            "Internal communications reveal discussions about strategies to manage public perception and regulatory scrutiny during the California energy crisis. Topics include price manipulation allegations and Enron's role in market dynamics.",
        keywords: ['California', 'energy crisis', 'regulation', 'PR'],
        dateGenerated: '2001-05-15',
        sourceEmails: 230,
    },
];

interface SummarizationClientProps {
    data: SummarizedEmail[];
    pageCount: number;
}

export default function SummarizationClient({ data, pageCount }: SummarizationClientProps) {
    const { keywords, dateRange } = useFilterStore();
    const [selectedEmail, setSelectedEmail] = useState<SummarizedEmail | null>(null);
    const [isDetailsOpen, setIsDetailsOpen] = useState(false);
    const [rowAction, setRowAction] = useState<DataTableRowAction<SummarizedEmail> | null>(null);

    const columns = useMemo(
        () =>
            getSummarizedEmailTableColumns({
                setRowAction,
                onViewDetails: (email: SummarizedEmail) => {
                    setSelectedEmail(email);
                    setIsDetailsOpen(true);
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
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>Email Summaries</CardTitle>
                    <CardDescription>
                        Explore AI-generated summaries focusing on pertinent
                        details from the Enron email dataset.
                        {(keywords || dateRange?.from) &&
                            ` Filtering by: ${keywords ? `keywords "${keywords}"` : ''
                            }${keywords && dateRange?.from ? ' and ' : ''}${dateRange?.from ? `dates` : ''
                            }.`}
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {placeholderSummariesData.length > 0 ? (
                        <Accordion type="single" collapsible className="w-full">
                            {placeholderSummariesData.map((summary, index) => (
                                <AccordionItem
                                    value={`item-${index}`}
                                    key={summary.id}
                                >
                                    <AccordionTrigger>
                                        <div className="flex items-center gap-3">
                                            <FileTextIcon className="h-5 w-5 text-primary" />
                                            <span className="font-medium">
                                                {summary.title}
                                            </span>
                                        </div>
                                    </AccordionTrigger>
                                    <AccordionContent className="space-y-3 pl-8">
                                        <p className="text-muted-foreground">
                                            {summary.content}
                                        </p>
                                        <div className="text-xs text-muted-foreground space-y-1">
                                            <p>
                                                <strong>Date Generated:</strong>{' '}
                                                {summary.dateGenerated}
                                            </p>
                                            <p>
                                                <strong>
                                                    Source Emails Analyzed:
                                                </strong>{' '}
                                                {summary.sourceEmails}
                                            </p>
                                            <div className="flex items-center gap-2 pt-1">
                                                <strong>Keywords:</strong>
                                                {summary.keywords.map(
                                                    (keyword) => (
                                                        <Badge
                                                            variant="secondary"
                                                            key={keyword}
                                                        >
                                                            {keyword}
                                                        </Badge>
                                                    )
                                                )}
                                            </div>
                                        </div>
                                    </AccordionContent>
                                </AccordionItem>
                            ))}
                        </Accordion>
                    ) : (
                        <p className="text-muted-foreground text-center py-8">
                            No summaries match your current filters.
                        </p>
                    )}
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Summarized Emails</CardTitle>
                    <CardDescription>
                        View and explore the summarized email data with advanced filtering and sorting capabilities.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <DataTableToolbar table={table}>
                        <DataTableFilterMenu
                            table={table}
                            shallow={shallow}
                            debounceMs={debounceMs}
                            throttleMs={throttleMs}
                        />
                        <DataTableSortList table={table} align="end" />
                    </DataTableToolbar>
                    <DataTable table={table} />
                </CardContent>
            </Card>

            <SummarizedEmailDetailsDialog
                email={selectedEmail}
                open={isDetailsOpen}
                onOpenChange={setIsDetailsOpen}
            />
        </div>
    );
}
