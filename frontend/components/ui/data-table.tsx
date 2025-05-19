'use client';

import * as React from 'react';
import {
	DndContext,
	KeyboardSensor,
	MouseSensor,
	TouchSensor,
	closestCenter,
	useSensor,
	useSensors,
	DragEndEvent,
	UniqueIdentifier,
} from '@dnd-kit/core';
import { restrictToVerticalAxis } from '@dnd-kit/modifiers';
import {
	SortableContext,
	arrayMove,
	useSortable,
	verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import {
	ColumnDef,
	ColumnFiltersState,
	Row,
	SortingState,
	VisibilityState,
	flexRender,
	getCoreRowModel,
	getFacetedRowModel,
	getFacetedUniqueValues,
	getFilteredRowModel,
	getPaginationRowModel,
	getSortedRowModel,
	useReactTable,
} from '@tanstack/react-table';
import {
	CheckCircle2Icon,
	ChevronDownIcon,
	ChevronLeftIcon,
	ChevronRightIcon,
	ChevronsLeftIcon,
	ChevronsRightIcon,
	ColumnsIcon,
	GripVerticalIcon,
	LoaderIcon,
	MoreVerticalIcon,
	PlusIcon,
} from 'lucide-react';
import { Area, AreaChart, CartesianGrid, XAxis } from 'recharts';
import { toast } from 'sonner';
import { useIsMobile } from '@/hooks/use-mobile'; // Assuming this exists
import { Badge } from '@/components/ui/badge'; // Assuming these exist
import { Button } from '@/components/ui/button'; // Assuming these exist
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from '@/components/ui/chart'; // Assuming these exist
import { Checkbox } from '@/components/ui/checkbox'; // Assuming this exists
import {
	DropdownMenu,
	DropdownMenuCheckboxItem,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'; // Assuming these exist
import { Input } from '@/components/ui/input'; // Assuming these exist
import { Label } from '@/components/ui/label'; // Assuming these exist
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from '@/components/ui/select'; // Assuming these exist
import {
	Sheet,
	SheetClose,
	SheetContent,
	SheetDescription,
	SheetFooter,
	SheetHeader,
	SheetTitle,
	SheetTrigger,
} from '@/components/ui/sheet'; // Assuming these exist
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from '@/components/ui/table'; // Assuming these exist
import { Tabs, TabsContent } from '@/components/ui/tabs'; // Assuming these exist
import { ProcessedEmail } from '@/db/schema';

// Create a separate component for the drag handle
function DragHandle({ id }: { id: UniqueIdentifier }) {
	// Changed id type
	const { attributes, listeners } = useSortable({
		id,
	});

	return (
		<Button
			{...attributes}
			{...listeners}
			variant="ghost"
			size="icon"
			className="size-7 text-muted-foreground hover:bg-transparent"
		>
			<GripVerticalIcon className="size-3 text-muted-foreground" />
			<span className="sr-only">Drag to reorder</span>
		</Button>
	);
}

const columns: ColumnDef<ProcessedEmail>[] = [
	{
		id: 'drag',
		header: () => null,
		cell: ({ row }) => <DragHandle id={row.id} />, // Use row.id directly
		enableSorting: false,
		enableHiding: false,
	},
	{
		id: 'select',
		header: ({ table }) => (
			<div className="flex items-center justify-center">
				<Checkbox
					checked={table.getIsAllPageRowsSelected()}
					onCheckedChange={(value) =>
						table.toggleAllPageRowsSelected(!!value)
					}
					aria-label="Select all"
				/>
			</div>
		),
		cell: ({ row }) => (
			<div className="flex items-center justify-center">
				<Checkbox
					checked={row.getIsSelected()}
					onCheckedChange={(value) => row.toggleSelected(!!value)}
					aria-label="Select row"
				/>
			</div>
		),
		enableSorting: false,
		enableHiding: false,
	},
	{
		accessorKey: 'messageId',
		header: 'Message ID',
		cell: ({ row }) => {
			return <TableCellViewer item={row.original} />;
		},
		enableHiding: false,
	},
	{
		accessorKey: 'type',
		header: 'Section Type',
		cell: ({ row }) => (
			<div className="w-32">
				<Badge
					variant="outline"
					className="px-1.5 text-muted-foreground"
				>
					{row.original.type}
				</Badge>
			</div>
		),
	},
	{
		accessorKey: 'limit',
		header: () => <div className="w-full text-right">Limit</div>,
		cell: ({ row }) => (
			<form
				onSubmit={(e) => {
					e.preventDefault();
					toast.promise(
						new Promise((resolve) => setTimeout(resolve, 1000)),
						{
							loading: `Saving ${row.original.subject}`, // Use a relevant field
							success: 'Done',
							error: 'Error',
						},
					);
				}}
			>
				<Label htmlFor={`limit-${row.id}`} className="sr-only">
					Limit
				</Label>
				<Input
					className="h-8 w-16 border-transparent bg-transparent text-right shadow-none hover:bg-input/30 focus-visible:border focus-visible:bg-background"
					defaultValue={row.original.limit}
					id={`limit-${row.id}`}
				/>
			</form>
		),
	},
	{
		accessorKey: 'reviewer',
		header: 'Reviewer',
		cell: ({ row }) => {
			const reviewer = row.original.reviewer;

			if (reviewer && reviewer !== 'Assign reviewer') {
				return reviewer;
			}

			return (
				<>
					<Label htmlFor={`reviewer-${row.id}`} className="sr-only">
						Reviewer
					</Label>
					<Select>
						<SelectTrigger
							className="h-8 w-40"
							id={`reviewer-${row.id}`}
						>
							<SelectValue placeholder="Assign reviewer" />
						</SelectTrigger>
						<SelectContent align="end">
							<SelectItem value="Eddie Lake">
								Eddie Lake
							</SelectItem>
							<SelectItem value="Jamik Tashpulatov">
								Jamik Tashpulatov
							</SelectItem>
						</SelectContent>
					</Select>
				</>
			);
		},
	},
	{
		id: 'actions',
		cell: () => (
			<DropdownMenu>
				<DropdownMenuTrigger asChild>
					<Button
						variant="ghost"
						className="flex size-8 text-muted-foreground data-[state=open]:bg-muted"
						size="icon"
					>
						<MoreVerticalIcon />
						<span className="sr-only">Open menu</span>
					</Button>
				</DropdownMenuTrigger>
				<DropdownMenuContent align="end" className="w-32">
					<DropdownMenuItem>Edit</DropdownMenuItem>
					<DropdownMenuItem>Make a copy</DropdownMenuItem>
					<DropdownMenuItem>Favorite</DropdownMenuItem>
					<DropdownMenuSeparator />
					<DropdownMenuItem>Delete</DropdownMenuItem>
				</DropdownMenuContent>
			</DropdownMenu>
		),
	},
];

function DraggableRow({ row }: { row: Row<ProcessedEmail> }) {
	const { transform, transition, setNodeRef, isDragging } = useSortable({
		id: row.id, // Use the row.id
	});

	return (
		<TableRow
			data-state={row.getIsSelected() && 'selected'}
			data-dragging={isDragging}
			ref={setNodeRef}
			className="relative z-0 data-[dragging=true]:z-10 data-[dragging=true]:opacity-80"
			style={{
				transform: CSS.Transform.toString(transform),
				transition: transition,
			}}
		>
			{row.getVisibleCells().map((cell) => (
				<TableCell key={cell.id}>
					{flexRender(cell.column.columnDef.cell, cell.getContext())}
				</TableCell>
			))}
		</TableRow>
	);
}

export function DataTable({ data: initialData }: { data: ProcessedEmail[] }) {
	const [data, setData] = React.useState<ProcessedEmail[]>(() => initialData);
	const [rowSelection, setRowSelection] = React.useState({});
	const [columnVisibility, setColumnVisibility] =
		React.useState<VisibilityState>({});
	const [columnFilters, setColumnFilters] =
		React.useState<ColumnFiltersState>([]);
	const [sorting, setSorting] = React.useState<SortingState>([]);
	const [pagination, setPagination] = React.useState({
		pageIndex: 0,
		pageSize: 10,
	});
	const sortableId = React.useId();
	const sensors = useSensors(
		useSensor(MouseSensor, {}),
		useSensor(TouchSensor, {}),
		useSensor(KeyboardSensor, {}),
	);

	const dataIds = React.useMemo<UniqueIdentifier[]>(() => {
		return data.map((item) => item.messageId ?? '');
	}, [data]);

	const table = useReactTable({
		data,
		columns,
		state: {
			sorting,
			columnVisibility,
			rowSelection,
			columnFilters,
			pagination,
		},
		getRowId: (row) => row.messageId ?? '', // Use messageId as the ID
		enableRowSelection: true,
		onRowSelectionChange: setRowSelection,
		onSortingChange: setSorting,
		onColumnFiltersChange: setColumnFilters,
		onColumnVisibilityChange: setColumnVisibility,
		onPaginationChange: setPagination,
		getCoreRowModel: getCoreRowModel(),
		getFilteredRowModel: getFilteredRowModel(),
		getPaginationRowModel: getPaginationRowModel(),
		getSortedRowModel: getSortedRowModel(),
		getFacetedRowModel: getFacetedRowModel(),
		getFacetedUniqueValues: getFacetedUniqueValues(),
	});

	function handleDragEnd(event: DragEndEvent) {
		const { active, over } = event;
		if (active && over && active.id !== over.id) {
			setData((currentData) => {
				const oldIndex = currentData.findIndex(
					(item) => item.messageId === active.id,
				);
				const newIndex = currentData.findIndex(
					(item) => item.messageId === over.id,
				);
				return arrayMove(currentData, oldIndex, newIndex);
			});
		}
	}

	return (
		<Tabs
			defaultValue="outline"
			className="flex w-full flex-col justify-start gap-6"
		>
			<div className="flex items-center justify-between px-4 lg:px-6">
				<div className="flex items-center gap-2">
					<DropdownMenu>
						<DropdownMenuTrigger asChild>
							<Button variant="outline" size="sm">
								<ColumnsIcon />
								<span className="hidden lg:inline">
									Customize Columns
								</span>
								<span className="lg:hidden">Columns</span>
								<ChevronDownIcon />
							</Button>
						</DropdownMenuTrigger>
						<DropdownMenuContent align="end" className="w-56">
							{table
								.getAllColumns()
								.filter(
									(column) =>
										typeof column.accessorFn !==
											'undefined' && column.getCanHide(),
								)
								.map((column) => {
									return (
										<DropdownMenuCheckboxItem
											key={column.id}
											className="capitalize"
											checked={column.getIsVisible()}
											onCheckedChange={(value) =>
												column.toggleVisibility(!!value)
											}
										>
											{column.id}
										</DropdownMenuCheckboxItem>
									);
								})}
						</DropdownMenuContent>
					</DropdownMenu>
					<Button variant="outline" size="sm">
						<PlusIcon />
						<span className="hidden lg:inline">Add Section</span>
					</Button>
				</div>
			</div>
			<TabsContent
				value="outline"
				className="relative flex flex-col gap-4 overflow-auto px-4 lg:px-6"
			>
				<div className="overflow-hidden rounded-lg border">
					<DndContext
						collisionDetection={closestCenter}
						modifiers={[restrictToVerticalAxis]}
						onDragEnd={handleDragEnd}
						sensors={sensors}
						id={sortableId}
					>
						<Table>
							<TableHeader className="sticky top-0 z-10 bg-muted">
								{table.getHeaderGroups().map((headerGroup) => (
									<TableRow key={headerGroup.id}>
										{headerGroup.headers.map((header) => {
											return (
												<TableHead
													key={header.id}
													colSpan={header.colSpan}
												>
													{header.isPlaceholder
														? null
														: flexRender(
																header.column
																	.columnDef
																	.header,
																header.getContext(),
														  )}
												</TableHead>
											);
										})}
									</TableRow>
								))}
							</TableHeader>
							<TableBody className="**:data-[slot=table-cell]:first:w-8">
								{table.getRowModel().rows?.length ? (
									<SortableContext
										items={dataIds}
										strategy={verticalListSortingStrategy}
									>
										{table.getRowModel().rows.map((row) => (
											<DraggableRow
												key={row.id}
												row={row}
											/>
										))}
									</SortableContext>
								) : (
									<TableRow>
										<TableCell
											colSpan={columns.length}
											className="h-24 text-center"
										>
											No results.
										</TableCell>
									</TableRow>
								)}
							</TableBody>
						</Table>
					</DndContext>
				</div>
				<div className="flex items-center justify-between px-4">
					<div className="hidden flex-1 text-sm text-muted-foreground lg:flex">
						{table.getFilteredSelectedRowModel().rows.length} of{' '}
						{table.getFilteredRowModel().rows.length} row(s)
						selected.
					</div>
					<div className="flex w-full items-center gap-8 lg:w-fit">
						<div className="hidden items-center gap-2 lg:flex">
							<Label
								htmlFor="rows-per-page"
								className="text-sm font-medium"
							>
								Rows per page
							</Label>
							<Select
								value={`${
									table.getState().pagination.pageSize
								}`}
								onValueChange={(value) => {
									table.setPageSize(Number(value));
								}}
							>
								<SelectTrigger
									className="w-20"
									id="rows-per-page"
								>
									<SelectValue
										placeholder={
											table.getState().pagination.pageSize
										}
									/>
								</SelectTrigger>
								<SelectContent side="top">
									{[10, 20, 30, 40, 50].map((pageSize) => (
										<SelectItem
											key={pageSize}
											value={`${pageSize}`}
										>
											{pageSize}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
						<div className="flex w-fit items-center justify-center text-sm font-medium">
							Page {table.getState().pagination.pageIndex + 1} of{' '}
							{table.getPageCount()}
						</div>
						<div className="ml-auto flex items-center gap-2 lg:ml-0">
							<Button
								variant="outline"
								className="hidden h-8 w-8 p-0 lg:flex"
								onClick={() => table.setPageIndex(0)}
								disabled={!table.getCanPreviousPage()}
							>
								<span className="sr-only">
									Go to first page
								</span>
								<ChevronsLeftIcon />
							</Button>
							<Button
								variant="outline"
								className="size-8"
								size="icon"
								onClick={() => table.previousPage()}
								disabled={!table.getCanPreviousPage()}
							>
								<span className="sr-only">
									Go to previous page
								</span>
								<ChevronLeftIcon />
							</Button>
							<Button
								variant="outline"
								className="size-8"
								size="icon"
								onClick={() => table.nextPage()}
								disabled={!table.getCanNextPage()}
							>
								<span className="sr-only">Go to next page</span>
								<ChevronRightIcon />
							</Button>
							<Button
								variant="outline"
								className="hidden size-8 lg:flex"
								size="icon"
								onClick={() =>
									table.setPageIndex(table.getPageCount() - 1)
								}
								disabled={!table.getCanNextPage()}
							>
								<span className="sr-only">Go to last page</span>
								<ChevronsRightIcon />
							</Button>
						</div>
					</div>
				</div>
			</TabsContent>
			<TabsContent
				value="past-performance"
				className="flex flex-col px-4 lg:px-6"
			>
				<div className="aspect-video w-full flex-1 rounded-lg border border-dashed"></div>
			</TabsContent>
			<TabsContent
				value="key-personnel"
				className="flex flex-col px-4 lg:px-6"
			>
				<div className="aspect-video w-full flex-1 rounded-lg border border-dashed"></div>
			</TabsContent>
			<TabsContent
				value="focus-documents"
				className="flex flex-col px-4 lg:px-6"
			>
				<div className="aspect-video w-full flex-1 rounded-lg border border-dashed"></div>
			</TabsContent>
		</Tabs>
	);
}

const chartData = [
	{ month: 'January', desktop: 186, mobile: 80 },
	{ month: 'February', desktop: 305, mobile: 200 },
	{ month: 'March', desktop: 237, mobile: 120 },
	{ month: 'April', desktop: 73, mobile: 190 },
	{ month: 'May', desktop: 209, mobile: 130 },
	{ month: 'June', desktop: 214, mobile: 140 },
];

const chartConfig = {
	desktop: {
		label: 'Desktop',
		color: 'var(--primary)',
	},
	mobile: {
		label: 'Mobile',
		color: 'var(--primary)',
	},
} satisfies ChartConfig;

function TableCellViewer({ item }: { item: ProcessedEmail }) {
	const isMobile = useIsMobile();

	return (
		<Sheet>
			<SheetTrigger asChild>
				<Button
					variant="link"
					className="w-fit px-0 text-left text-foreground"
				>
					{item.messageId}
				</Button>
			</SheetTrigger>
			<SheetContent side="right" className="flex flex-col">
				<SheetHeader className="gap-1">
					<SheetTitle>{item.messageId}</SheetTitle>
					<SheetDescription>
						Showing details for Message ID: {item.messageId}
					</SheetDescription>
				</SheetHeader>
				<div className="flex flex-1 flex-col gap-4 overflow-y-auto py-4 text-sm">
					{!isMobile && (
						<>
							<ChartContainer config={chartConfig}>
								<AreaChart
									accessibilityLayer
									data={chartData}
									margin={{
										left: 0,
										right: 10,
									}}
								>
									<CartesianGrid vertical={false} />
									<XAxis
										dataKey="month"
										tickLine={false}
										axisLine={false}
										tickMargin={8}
										tickFormatter={(value) =>
											value.slice(0, 3)
										}
										hide
									/>
									<ChartTooltip
										cursor={false}
										content={
											<ChartTooltipContent indicator="dot" />
										}
									/>
									<Area
										dataKey="mobile"
										type="natural"
										fill="var(--color-mobile)"
										fillOpacity={0.6}
										stroke="var(--color-mobile)"
										stackId="a"
									/>
									<Area
										dataKey="desktop"
										type="natural"
										fill="var(--color-desktop)"
										fillOpacity={0.4}
										stroke="var(--color-desktop)"
										stackId="a"
									/>
								</AreaChart>
							</ChartContainer>
						</>
					)}
					<form className="flex flex-col gap-4">
						<div className="flex flex-col gap-3">
							<Label htmlFor="messageId">Message ID</Label>
							<Input
								id="messageId"
								defaultValue={item.messageId}
							/>
						</div>
						<div className="flex flex-col gap-3">
							<Label htmlFor="subject">Subject</Label>
							<Input id="subject" defaultValue={item.subject} />
						</div>
						<div className="grid grid-cols-2 gap-4">
							<div className="flex flex-col gap-3">
								<Label htmlFor="target">Target</Label>
								<Input id="target" defaultValue={item.limit} />
							</div>
							<div className="flex flex-col gap-3">
								<Label htmlFor="limit">Limit</Label>
								<Input id="limit" defaultValue={item.limit} />
							</div>
						</div>
					</form>
				</div>
				<SheetFooter className="mt-auto flex gap-2 sm:flex-col sm:space-x-0">
					<Button className="w-full">Submit</Button>
					<SheetClose asChild>
						<Button variant="outline" className="w-full">
							Done
						</Button>
					</SheetClose>
				</SheetFooter>
			</SheetContent>
		</Sheet>
	);
}
