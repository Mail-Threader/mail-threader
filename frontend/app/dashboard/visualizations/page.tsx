'use client';

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
	ChartLegend,
	ChartLegendContent,
	type ChartConfig,
} from '@/components/ui/chart';
import Image from 'next/image';
import {
	Bar as ClientBar,
	BarChart as ClientBarChart,
	CartesianGrid as ClientCartesianGrid,
	XAxis as ClientXAxis,
	YAxis as ClientYAxis,
	Line as ClientLine,
	LineChart as ClientLineChart,
	Pie as ClientPie,
	PieChart as ClientPieChart,
	Cell as ClientCell,
	ResponsiveContainer as ClientResponsiveContainer,
} from 'recharts';
import { useFilterStore } from '@/store/filter-store';
import { useMemo } from 'react';

const initialEmailVolumeData = [
	{ month: "Jan '00", date: '2000-01-01', desktop: 186, mobile: 80 },
	{ month: "Feb '00", date: '2000-02-01', desktop: 305, mobile: 200 },
	{ month: "Mar '00", date: '2000-03-01', desktop: 237, mobile: 120 },
	{ month: "Apr '00", date: '2000-04-01', desktop: 73, mobile: 190 },
	{ month: "May '00", date: '2000-05-01', desktop: 209, mobile: 130 },
	{ month: "Jun '00", date: '2000-06-01', desktop: 214, mobile: 140 },
	{ month: "Jul '00", date: '2000-07-01', desktop: 320, mobile: 150 },
	{ month: "Aug '00", date: '2000-08-01', desktop: 280, mobile: 160 },
	{ month: "Sep '00", date: '2000-09-01', desktop: 250, mobile: 170 },
	{ month: "Oct '00", date: '2000-10-01', desktop: 450, mobile: 180 },
	{ month: "Nov '00", date: '2000-11-01', desktop: 380, mobile: 190 },
	{ month: "Dec '00", date: '2000-12-01', desktop: 500, mobile: 200 },
];

const emailVolumeChartConfig: ChartConfig = {
	// Ensure ChartConfig is typed correctly or use 'as const'
	desktop: { label: 'Internal Emails', color: 'hsl(var(--chart-1))' },
	mobile: { label: 'External Emails', color: 'hsl(var(--chart-2))' },
};

const initialTopSendersData = [
	{ name: 'J.Skilling', emails: 4500, fill: 'hsl(var(--chart-1))' },
	{ name: 'A.Fastow', emails: 3200, fill: 'hsl(var(--chart-2))' },
	{ name: 'K.Lay', emails: 2800, fill: 'hsl(var(--chart-3))' },
	{ name: 'R.Mark', emails: 2100, fill: 'hsl(var(--chart-4))' },
	{ name: 'L.Pai', emails: 1800, fill: 'hsl(var(--chart-5))' },
];

const initialSentimentData = [
	{ name: 'Positive', value: 400, fill: 'hsl(var(--chart-1))' },
	{ name: 'Negative', value: 300, fill: 'hsl(var(--chart-2))' },
	{ name: 'Neutral', value: 300, fill: 'hsl(var(--chart-3))' },
];

export default function VisualizationsPage() {
	const { keywords, dateRange } = useFilterStore();

	// Placeholder for actual data filtering based on keywords and dateRange
	// For now, charts will use initialData. Filtering logic would be complex for charts.
	const emailVolumeData = initialEmailVolumeData;
	const topSendersData = initialTopSendersData;
	const sentimentData = initialSentimentData;

	const activeFilterMessage = useMemo(() => {
		if (keywords || dateRange?.from) {
			return ` Chart data would be filtered by: ${
				keywords ? `keywords "${keywords}"` : ''
			}${keywords && dateRange?.from ? ' and ' : ''}${
				dateRange?.from ? `selected dates` : ''
			}.`;
		}
		return 'Displaying general visualizations.';
	}, [keywords, dateRange]);

	return (
		<div className="grid gap-6 lg:grid-cols-2">
			<Card className="lg:col-span-2">
				<CardHeader>
					<CardTitle>Visualizations Overview</CardTitle>
					<CardDescription>{activeFilterMessage}</CardDescription>
				</CardHeader>
			</Card>
			<Card>
				<CardHeader>
					<CardTitle>Email Volume Over Time</CardTitle>
					<CardDescription>
						Monthly internal vs. external email communication.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<ChartContainer
						config={emailVolumeChartConfig}
						className="h-[300px] w-full"
					>
						<ClientLineChart
							accessibilityLayer
							data={emailVolumeData}
							margin={{ left: 12, right: 12 }}
						>
							<ClientCartesianGrid vertical={false} />
							<ClientXAxis
								dataKey="month"
								tickLine={false}
								axisLine={false}
								tickMargin={8}
								tickFormatter={(value) => value.slice(0, 3)}
							/>
							<ClientYAxis />
							<ChartTooltip
								cursor={false}
								content={<ChartTooltipContent hideLabel />}
							/>
							<ChartLegend content={<ChartLegendContent />} />
							<ClientLine
								dataKey="desktop"
								type="monotone"
								stroke="var(--color-desktop)"
								strokeWidth={2}
								dot={false}
							/>
							<ClientLine
								dataKey="mobile"
								type="monotone"
								stroke="var(--color-mobile)"
								strokeWidth={2}
								dot={false}
							/>
						</ClientLineChart>
					</ChartContainer>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Top Email Senders</CardTitle>
					<CardDescription>
						Number of emails sent by top individuals.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<ChartContainer config={{}} className="h-[300px] w-full">
						<ClientBarChart
							accessibilityLayer
							data={topSendersData}
							layout="vertical"
							margin={{ left: 12, right: 12 }}
						>
							<ClientCartesianGrid horizontal={false} />
							<ClientYAxis
								dataKey="name"
								type="category"
								tickLine={false}
								axisLine={false}
								tickMargin={8}
							/>
							<ClientXAxis dataKey="emails" type="number" />
							<ChartTooltip
								cursor={false}
								content={<ChartTooltipContent />}
							/>
							<ClientBar dataKey="emails" radius={4}>
								{topSendersData.map((entry, index) => (
									<ClientCell
										key={`cell-${index}`}
										fill={entry.fill}
									/>
								))}
							</ClientBar>
						</ClientBarChart>
					</ChartContainer>
				</CardContent>
			</Card>

			<Card className="lg:col-span-2">
				<CardHeader>
					<CardTitle>Email Sentiment Analysis</CardTitle>
					<CardDescription>
						Distribution of email sentiment across the dataset.
					</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col items-center gap-4 sm:flex-row">
					<div className="flex-1">
						<ChartContainer
							config={{}}
							className="h-[250px] w-full sm:w-[250px]"
						>
							<ClientPieChart accessibilityLayer>
								<ChartTooltip
									content={<ChartTooltipContent hideLabel />}
								/>
								<ClientPie
									data={sentimentData}
									dataKey="value"
									nameKey="name"
									cx="50%"
									cy="50%"
									outerRadius={80}
								>
									{sentimentData.map((entry, index) => (
										<ClientCell
											key={`cell-${index}`}
											fill={entry.fill}
										/>
									))}
								</ClientPie>
								<ChartLegend content={<ChartLegendContent />} />
							</ClientPieChart>
						</ChartContainer>
					</div>
					<div className="flex-1 space-y-2">
						<p className="text-sm text-muted-foreground">
							This chart illustrates the overall sentiment
							expressed in the Enron email dataset. Understanding
							sentiment can provide insights into the corporate
							culture, employee morale, and reactions to specific
							events.
						</p>
						<p className="text-sm text-muted-foreground">
							Further drill-down capabilities could reveal
							sentiment trends over time or sentiment associated
							with key individuals or topics.
						</p>
					</div>
				</CardContent>
			</Card>

			<Card className="lg:col-span-2">
				<CardHeader>
					<CardTitle>Communication Network</CardTitle>
					<CardDescription>
						Visualizing key communicators and their connections
						(placeholder).
					</CardDescription>
				</CardHeader>
				<CardContent className="flex justify-center items-center">
					<Image
						src="https://placehold.co/800x400.png"
						alt="Communication Network Graph"
						width={800}
						height={400}
						className="rounded-lg shadow-md w-full h-auto object-cover"
						data-ai-hint="network graph"
					/>
				</CardContent>
			</Card>
		</div>
	);
}
