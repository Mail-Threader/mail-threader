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
import { useEffect, useMemo, useState } from 'react';
import {
	getSentimentData,
	getTopEmailSenders,
	getVisualizationImagesLinks,
} from '@/actions/data';

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

// Define colors for sentiment pie chart for POSITIVE, NEGATIVE, NEUTRAL
const sentimentPieChartColors = ['#4CAF50', '#F44336', '#FF9800'];

export default function VisualizationsPage() {
	const [topEmailSendersData, setTopEmailSendersData] = useState<
		{
			sender: string | null;
			count: number;
		}[]
	>([]);
	const [sentimentData, setSentimentData] = useState<
		{
			sentiment: string | null;
			count: number;
		}[]
	>([]);
	const [visualizationsImages, setVisualizationsImages] = useState<
		{
			name: string | null;
			url: string | null;
		}[]
	>([]);

	const { keywords, dateRange } = useFilterStore();

	// Placeholder for actual data filtering based on keywords and dateRange
	// For now, charts will use initialData. Filtering logic would be complex for charts.
	const emailVolumeData = initialEmailVolumeData;

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

	useEffect(() => {
		const fetchData = async () => {
			const topSendersRes = await getTopEmailSenders();
			const sentimentRes = await getSentimentData();
			const visualizationImagesRes = await getVisualizationImagesLinks();
			if (topSendersRes.length > 0) {
				setTopEmailSendersData(topSendersRes);
			}
			if (sentimentRes.length > 0) {
				setSentimentData(sentimentRes);
			}
			if (visualizationImagesRes.length > 0) {
				setVisualizationsImages(visualizationImagesRes);
			}
		};
		fetchData();
	}, []);

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
								stroke="black"
								strokeWidth={2}
								dot={false}
							/>
							<ClientLine
								dataKey="mobile"
								type="monotone"
								stroke="red"
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
							data={topEmailSendersData}
							layout="vertical"
							margin={{ left: 12, right: 12 }}
						>
							<ClientCartesianGrid horizontal={false} />
							<ClientYAxis
								dataKey="sender"
								type="category"
								tickLine={false}
								axisLine={false}
								tickMargin={8}
							/>
							<ClientXAxis dataKey="count" type="number" />
							<ChartTooltip
								cursor={false}
								content={<ChartTooltipContent />}
							/>
							<ClientBar dataKey="count" radius={4}>
								{topEmailSendersData.map((entry, index) => (
									<ClientCell
										key={`cell-${index}`}
										fill={`hsl(var(--chart-${index + 1}))`}
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
									dataKey="count"
									nameKey="sentiment"
									cx="50%"
									cy="50%"
									outerRadius={80}
								>
									{sentimentData.map((entry, index) => (
										<ClientCell
											key={`cell-${index}`}
											fill={
												sentimentPieChartColors[index]
											}
											stroke="white"
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
					<CardTitle>Generated Visualizations</CardTitle>
					<CardDescription>
						Visualizing key communicators and their connections.
					</CardDescription>
				</CardHeader>
				<CardContent className="flex gap-4 flex-wrap">
					{visualizationsImages.map((imageObj, index) => (
						<Image
							key={index}
							src={imageObj.url || ''}
							alt={imageObj.name || 'Visualization'}
							width={500}
							height={300}
							quality={100}
							className="rounded-lg shadow-md w-full h-auto object-cover"
							data-ai-hint={imageObj.name || 'network graph'}
						/>
					))}
				</CardContent>
			</Card>
		</div>
	);
}
