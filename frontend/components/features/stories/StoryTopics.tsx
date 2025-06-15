import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

interface StoryTopicsProps {
    topics: Array<{
        name: string;
        count: number;
        sentiment: number;
        keywords: string[];
    }>;
}

export function StoryTopics({ topics }: StoryTopicsProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-lg">Topic Analysis</CardTitle>
                <p className="text-sm text-muted-foreground">
                    Key topics and their evolution
                </p>
            </CardHeader>
            <CardContent>
                <div className="space-y-6">
                    {topics.map((topic, index) => (
                        <div key={index} className="space-y-2">
                            <div className="flex items-center justify-between">
                                <div className="font-medium">{topic.name}</div>
                                <div className="text-sm text-muted-foreground">
                                    {topic.count} mentions
                                </div>
                            </div>
                            <Progress value={(topic.sentiment + 1) * 50} className="h-2" />
                            <div className="flex flex-wrap gap-2">
                                {topic.keywords.map((keyword, idx) => (
                                    <Badge key={idx} variant="secondary">
                                        {keyword}
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
