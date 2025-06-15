import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Timeline, TimelineItem } from "@/components/ui/timeline";
import { format } from "date-fns";

interface StoryTimelineProps {
    timeline: {
        startDate: string;
        endDate: string;
        events: Array<{
            date: string;
            description: string;
            significance: number;
        }>;
    };
}

export function StoryTimeline({ timeline }: StoryTimelineProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-lg">Timeline</CardTitle>
                <p className="text-sm text-muted-foreground">
                    {format(new Date(timeline.startDate), "MMM d, yyyy")} - {format(new Date(timeline.endDate), "MMM d, yyyy")}
                </p>
            </CardHeader>
            <CardContent>
                <Timeline>
                    {timeline.events.map((event, index) => (
                        <TimelineItem
                            key={index}
                            className={`relative ${event.significance > 0.7
                                ? "border-l-2 border-red-500"
                                : event.significance > 0.4
                                    ? "border-l-2 border-yellow-500"
                                    : "border-l-2 border-blue-500"
                                }`}
                        >
                            <div className="flex flex-col space-y-1">
                                <div className="text-sm font-medium">
                                    {format(new Date(event.date), "MMM d, yyyy")}
                                </div>
                                <div className="text-sm text-muted-foreground">
                                    {event.description}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                    Significance: {(event.significance * 100).toFixed(0)}%
                                </div>
                            </div>
                        </TimelineItem>
                    ))}
                </Timeline>
            </CardContent>
        </Card>
    );
}
