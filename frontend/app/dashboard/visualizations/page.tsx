
"use client"; // Required for Recharts/Chart components

import { Bar, BarChart, CartesianGrid, XAxis, YAxis, Line, LineChart, Pie, PieChart, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from "@/components/ui/chart";
import Image from "next/image";

const emailVolumeData = [
  { month: "Jan '00", desktop: 186, mobile: 80 },
  { month: "Feb '00", desktop: 305, mobile: 200 },
  { month: "Mar '00", desktop: 237, mobile: 120 },
  { month: "Apr '00", desktop: 73, mobile: 190 },
  { month: "May '00", desktop: 209, mobile: 130 },
  { month: "Jun '00", desktop: 214, mobile: 140 },
  { month: "Jul '00", desktop: 320, mobile: 150 },
  { month: "Aug '00", desktop: 280, mobile: 160 },
  { month: "Sep '00", desktop: 250, mobile: 170 },
  { month: "Oct '00", desktop: 450, mobile: 180 },
  { month: "Nov '00", desktop: 380, mobile: 190 },
  { month: "Dec '00", desktop: 500, mobile: 200 },
];

const emailVolumeChartConfig = {
  desktop: { label: "Internal Emails", color: "hsl(var(--chart-1))" },
  mobile: { label: "External Emails", color: "hsl(var(--chart-2))" },
} satisfies ChartConfig;

const topSendersData = [
  { name: "J.Skilling", emails: 4500, fill: "hsl(var(--chart-1))" },
  { name: "A.Fastow", emails: 3200, fill: "hsl(var(--chart-2))"  },
  { name: "K.Lay", emails: 2800, fill: "hsl(var(--chart-3))"  },
  { name: "R.Mark", emails: 2100, fill: "hsl(var(--chart-4))"  },
  { name: "L.Pai", emails: 1800, fill: "hsl(var(--chart-5))"  },
];

const sentimentData = [
  { name: 'Positive', value: 400, fill: 'hsl(var(--chart-1))' },
  { name: 'Negative', value: 300, fill: 'hsl(var(--chart-2))' },
  { name: 'Neutral', value: 300, fill: 'hsl(var(--chart-3))' },
];


export default function VisualizationsPage() {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Email Volume Over Time</CardTitle>
          <CardDescription>Monthly internal vs. external email communication.</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={emailVolumeChartConfig} className="h-[300px] w-full">
            <LineChart accessibilityLayer data={emailVolumeData} margin={{ left: 12, right: 12 }}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="month"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickFormatter={(value) => value.slice(0, 3)}
              />
              <YAxis />
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Line dataKey="desktop" type="monotone" stroke="var(--color-desktop)" strokeWidth={2} dot={false} />
              <Line dataKey="mobile" type="monotone" stroke="var(--color-mobile)" strokeWidth={2} dot={false} />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Top Email Senders</CardTitle>
          <CardDescription>Number of emails sent by top individuals.</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={{}} className="h-[300px] w-full">
            <BarChart accessibilityLayer data={topSendersData} layout="vertical" margin={{ left: 12, right: 12 }}>
              <CartesianGrid horizontal={false} />
              <YAxis dataKey="name" type="category" tickLine={false} axisLine={false} tickMargin={8} />
              <XAxis dataKey="emails" type="number" />
              <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
              <Bar dataKey="emails" radius={4}>
                {topSendersData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Email Sentiment Analysis</CardTitle>
          <CardDescription>Distribution of email sentiment across the dataset.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4 sm:flex-row">
          <div className="flex-1">
            <ChartContainer config={{}} className="h-[250px] w-full sm:w-[250px]">
              <PieChart accessibilityLayer>
                <ChartTooltip content={<ChartTooltipContent hideLabel />} />
                <Pie data={sentimentData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}>
                    {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                </Pie>
                <ChartLegend content={<ChartLegendContent />} />
              </PieChart>
            </ChartContainer>
          </div>
          <div className="flex-1 space-y-2">
            <p className="text-sm text-muted-foreground">
              This chart illustrates the overall sentiment expressed in the Enron email dataset.
              Understanding sentiment can provide insights into the corporate culture, employee morale,
              and reactions to specific events.
            </p>
            <p className="text-sm text-muted-foreground">
              Further drill-down capabilities could reveal sentiment trends over time or sentiment
              associated with key individuals or topics.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Communication Network</CardTitle>
          <CardDescription>Visualizing key communicators and their connections (placeholder).</CardDescription>
        </CardHeader>
        <CardContent className="flex justify-center items-center">
           <Image
            src="https://picsum.photos/800/400?random=3"
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
