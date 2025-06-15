import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { LightbulbIcon, Share2Icon, ThumbsUpIcon, ChevronDownIcon } from "lucide-react";
import { useState } from "react";
import { StoryMetrics } from "./StoryMetrics";
import { StoryTimeline } from "./StoryTimeline";
import { StoryNetwork } from "./StoryNetwork";
import { StoryTopics } from "./StoryTopics";
import { StoryEmails } from "./StoryEmails";

function getInitials(title: string) {
    return title
        .split(" ")
        .map((w) => w[0])
        .join("")
        .slice(0, 2)
        .toUpperCase();
}

interface StoryCardProps {
    story: {
        id: string;
        title: string;
        narrative: string;
        keyInsights: string[];
        relatedVisualizations: string[];
        dateDiscovered: string;
        tags: string[];
        metrics?: {
            influenceScore?: number;
            avgResponseTime?: number;
            emailCount?: number;
            peakActivity?: {
                day: string;
                count: number;
            };
        };
        timeline?: {
            startDate: string;
            endDate: string;
            events: Array<{
                date: string;
                description: string;
                significance: number;
            }>;
        };
        network?: {
            nodes: Array<{
                id: string;
                label: string;
                value: number;
            }>;
            links: Array<{
                source: string;
                target: string;
                value: number;
            }>;
        };
        topics?: Array<{
            name: string;
            count: number;
            sentiment: number;
            keywords: string[];
        }>;
        relatedEmails?: Array<{
            subject: string;
            date: string;
            from: string;
            to: string;
            bodyPreview: string;
            category: string;
        }>;
    };
}

export function StoryCard({ story }: StoryCardProps) {
    const [expanded, setExpanded] = useState(false);

    return (
        <Card className="flex flex-col shadow-md border border-gray-200 rounded-xl bg-white hover:shadow-lg transition-shadow p-4">
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <div className="flex items-center justify-center w-16 h-16 rounded-full bg-blue-100 text-blue-700 font-bold text-2xl">
                        {getInitials(story.title)}
                    </div>
                    <div className="text-xs text-gray-400">Discovered: {story.dateDiscovered}</div>
                </div>
                <CardTitle className="text-xl font-semibold mt-2 leading-tight">{story.title}</CardTitle>
            </CardHeader>
            <CardContent className="flex-grow space-y-4 pt-2">
                <p className="text-base text-gray-700 leading-relaxed mb-2">{story.narrative}</p>
                {story.metrics && <StoryMetrics metrics={story.metrics} />}
                {expanded && (
                    <>
                        {story.timeline && <StoryTimeline timeline={story.timeline} />}
                        {story.network && <StoryNetwork network={story.network} />}
                        {story.topics && <StoryTopics topics={story.topics} />}
                        {story.relatedEmails && <StoryEmails emails={story.relatedEmails} />}
                    </>
                )}
                <div className="flex items-center justify-between pt-2">
                    <div className="flex gap-1 flex-wrap">
                        {story.tags.slice(0, 3).map((tag) => (
                            <Badge variant="outline" key={tag} className="mb-1">
                                {tag}
                            </Badge>
                        ))}
                    </div>
                    <div className="flex gap-2">
                        <Button variant="ghost" size="icon" aria-label="Like story">
                            <ThumbsUpIcon className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="icon" aria-label="Share story">
                            <Share2Icon className="h-4 w-4" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setExpanded(!expanded)}
                            aria-label="Expand story"
                        >
                            <ChevronDownIcon className={`h-4 w-4 transition-transform ${expanded ? 'rotate-180' : ''}`} />
                        </Button>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
