import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ClockIcon, ActivityIcon, MailIcon, TrendingUpIcon } from "lucide-react";

interface StoryMetricsProps {
    metrics: {
        influenceScore?: number;
        avgResponseTime?: number;
        emailCount?: number;
        peakActivity?: {
            day: string;
            count: number;
        };
    };
}

export function StoryMetrics({ metrics }: StoryMetricsProps) {
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metrics.influenceScore !== undefined && (
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Influence Score</CardTitle>
                        <TrendingUpIcon className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{metrics.influenceScore.toFixed(2)}</div>
                        <Progress value={metrics.influenceScore * 100} className="mt-2" />
                    </CardContent>
                </Card>
            )}

            {metrics.avgResponseTime !== undefined && (
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
                        <ClockIcon className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{metrics.avgResponseTime.toFixed(1)}h</div>
                        <p className="text-xs text-muted-foreground">Average time to respond</p>
                    </CardContent>
                </Card>
            )}

            {metrics.emailCount !== undefined && (
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Emails</CardTitle>
                        <MailIcon className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{metrics.emailCount}</div>
                        <p className="text-xs text-muted-foreground">Emails sent/received</p>
                    </CardContent>
                </Card>
            )}

            {metrics.peakActivity && (
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Peak Activity</CardTitle>
                        <ActivityIcon className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{metrics.peakActivity.count}</div>
                        <p className="text-xs text-muted-foreground">Emails on {metrics.peakActivity.day}</p>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
