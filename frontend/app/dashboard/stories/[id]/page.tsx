'use client';

import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { StoryMetrics } from '@/components/features/stories/StoryMetrics';
import { StoryTimeline } from '@/components/features/stories/StoryTimeline';
import { StoryNetwork } from '@/components/features/stories/StoryNetwork';
import { StoryTopics } from '@/components/features/stories/StoryTopics';
import { StoryEmails } from '@/components/features/stories/StoryEmails';
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface Story {
    metadata: {
        generation_date: string;
        total_emails: number;
        time_span: {
            start: string;
            end: string;
        };
    };
    key_actors: {
        title: string;
        main_characters: Array<{
            name: string;
            role: string;
            metrics: Record<string, number>;
            influence_score: number;
        }>;
        relationships: Array<{
            participants: string[];
            strength: number;
            type: string;
        }>;
    };
    topic_evolution: {
        title: string;
        timeline: Array<{
            period: string;
            topics: Array<{
                id: string;
                count: number;
                keywords: string[];
            }>;
        }>;
        key_topics: Array<{
            id: string;
            keywords: string[];
            significance: number;
        }>;
    };
    significant_events: {
        title: string;
        events: Array<{
            date: string;
            significance: number;
            description: string;
            key_topics: string[];
            sample_subjects: string[];
        }>;
    };
    email_threads: {
        title: string;
        threads: Array<{
            subject: string;
            size: number;
            participants: string[];
            timeline: {
                start: string;
                end: string;
            };
        }>;
    };
    email_connections: {
        title: string;
        network_properties: {
            total_connections: number;
            connection_types: Record<string, number>;
            communities: number;
        };
        central_emails: Array<{
            email_id: string;
            centrality_score: number;
        }>;
        temporal_patterns: Record<string, any>;
    };
    narrative_summary: string;
}

export default function StoryDetailPage() {
    const params = useParams();
    const [story, setStory] = useState<Story | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStory = async () => {
            try {
                // TODO: Replace with actual API call
                const response = await fetch(`/api/stories/${params.id}`);
                const data = await response.json();
                setStory(data);
            } catch (error) {
                console.error('Error fetching story:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchStory();
    }, [params.id]);

    if (loading) {
        return <div className="flex items-center justify-center min-h-screen">Loading...</div>;
    }

    if (!story) {
        return <div className="flex items-center justify-center min-h-screen">Story not found</div>;
    }

    return (
        <div className="container mx-auto py-8 px-4">
            <div className="mb-8">
                <h1 className="text-3xl font-bold mb-4">{story.key_actors.title}</h1>
                <p className="text-slate-900 mb-4">{story.narrative_summary}</p>
                <div className="flex gap-2 mb-4">
                    <Badge variant="outline">Total Emails: {story.metadata.total_emails}</Badge>
                    <Badge variant="outline">
                        Time Span: {new Date(story.metadata.time_span.start).toLocaleDateString()} -{' '}
                        {new Date(story.metadata.time_span.end).toLocaleDateString()}
                    </Badge>
                </div>
            </div>

            <Tabs defaultValue="actors" className="w-full">
                <TabsList className="grid w-full grid-cols-5 mb-6">
                    <TabsTrigger value="actors">Key Actors</TabsTrigger>
                    <TabsTrigger value="topics">Topics</TabsTrigger>
                    <TabsTrigger value="events">Events</TabsTrigger>
                    <TabsTrigger value="threads">Threads</TabsTrigger>
                    <TabsTrigger value="network">Network</TabsTrigger>
                </TabsList>

                <TabsContent value="actors">
                    <Card>
                        <CardHeader>
                            <CardTitle>Key Actors</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-6 overflow-x-auto pb-2">
                                {story.key_actors.main_characters.map((actor) => (
                                    <Card key={actor.name} className="min-w-[280px] shadow-md border rounded-xl flex-shrink-0">
                                        <CardHeader>
                                            <CardTitle className="flex items-center justify-between">
                                                {actor.name}
                                                <TooltipProvider>
                                                    <Tooltip>
                                                        <TooltipTrigger asChild>
                                                            <span className="text-sm text-muted-foreground flex items-center gap-1 ml-4">
                                                                Influence: {actor.influence_score.toFixed(2)}
                                                                <Info className="h-4 w-4" />
                                                            </span>
                                                        </TooltipTrigger>
                                                        <TooltipContent>
                                                            <p>Influence score based on network centrality metrics</p>
                                                        </TooltipContent>
                                                    </Tooltip>
                                                </TooltipProvider>
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-2">
                                                {Object.entries(actor.metrics).map(([key, value]) => (
                                                    <div key={key} className="flex justify-between text-sm">
                                                        <span className="text-muted-foreground">{key}:</span>
                                                        <span className="font-medium">{value.toFixed(2)}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="topics">
                    <Card>
                        <CardHeader>
                            <CardTitle>Topic Evolution</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-8">
                                {story.topic_evolution.timeline.map((period) => (
                                    <div key={period.period} className="space-y-2">
                                        <h3 className="text-lg font-semibold mb-2">{period.period}</h3>
                                        <div className="flex gap-4 overflow-x-auto pb-2">
                                            {period.topics.map((topic) => (
                                                <Card key={topic.id} className="min-w-[220px] shadow border rounded-xl flex-shrink-0">
                                                    <CardHeader>
                                                        <CardTitle className="text-base">Topic {topic.id}</CardTitle>
                                                    </CardHeader>
                                                    <CardContent>
                                                        <div className="space-y-2">
                                                            <div className="flex flex-wrap gap-1">
                                                                {topic.keywords.map((keyword) => (
                                                                    <Badge key={keyword} variant="secondary">
                                                                        {keyword}
                                                                    </Badge>
                                                                ))}
                                                            </div>
                                                            <div className="text-sm text-muted-foreground">
                                                                Count: {topic.count}
                                                            </div>
                                                        </div>
                                                    </CardContent>
                                                </Card>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="events">
                    <Card>
                        <CardHeader>
                            <CardTitle>Significant Events</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-6 overflow-x-auto pb-2">
                                {story.significant_events.events.map((event, index) => (
                                    <Card key={index} className="min-w-[320px] shadow border rounded-xl flex-shrink-0">
                                        <CardHeader>
                                            <CardTitle className="text-base">
                                                {new Date(event.date).toLocaleDateString()}
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-2">
                                                <p>{event.description}</p>
                                                <div className="flex flex-wrap gap-1">
                                                    {event.key_topics.map((topic) => (
                                                        <Badge key={topic} variant="secondary">
                                                            {topic}
                                                        </Badge>
                                                    ))}
                                                </div>
                                                <div className="text-sm text-muted-foreground">
                                                    <p>Sample Subjects:</p>
                                                    <ul className="list-disc list-inside">
                                                        {event.sample_subjects.map((subject, i) => (
                                                            <li key={i}>{subject}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="threads">
                    <Card>
                        <CardHeader>
                            <CardTitle>Email Threads</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-6 overflow-x-auto pb-2">
                                {story.email_threads.threads.map((thread, index) => (
                                    <Card key={index} className="min-w-[320px] shadow border rounded-xl flex-shrink-0">
                                        <CardHeader>
                                            <CardTitle className="text-base">{thread.subject}</CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-2">
                                                <div className="flex justify-between text-sm text-muted-foreground">
                                                    <span>Size: {thread.size} emails</span>
                                                    <span>
                                                        {new Date(thread.timeline.start).toLocaleDateString()} -{' '}
                                                        {new Date(thread.timeline.end).toLocaleDateString()}
                                                    </span>
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium mb-1">Participants:</p>
                                                    <div className="flex flex-wrap gap-1">
                                                        {thread.participants.map((participant) => (
                                                            <Badge key={participant} variant="secondary">
                                                                {participant}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="network">
                    <Card>
                        <CardHeader>
                            <CardTitle>Network Analysis</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                    <Card className="shadow border rounded-xl">
                                        <CardHeader>
                                            <CardTitle className="text-base">Network Properties</CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-sm text-muted-foreground">Total Connections:</span>
                                                    <span className="text-sm font-medium">
                                                        {story.email_connections.network_properties.total_connections}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-sm text-muted-foreground">Communities:</span>
                                                    <span className="text-sm font-medium">
                                                        {story.email_connections.network_properties.communities}
                                                    </span>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>

                                    <Card className="shadow border rounded-xl">
                                        <CardHeader>
                                            <CardTitle className="text-base">Connection Types</CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-2">
                                                {Object.entries(story.email_connections.network_properties.connection_types).map(
                                                    ([type, count]) => (
                                                        <div key={type} className="flex justify-between">
                                                            <span className="text-sm text-muted-foreground">{type}:</span>
                                                            <span className="text-sm font-medium">{count}</span>
                                                        </div>
                                                    ),
                                                )}
                                            </div>
                                        </CardContent>
                                    </Card>

                                    <Card className="shadow border rounded-xl">
                                        <CardHeader>
                                            <CardTitle className="text-base">Central Emails</CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-2">
                                                {story.email_connections.central_emails.map((email) => (
                                                    <div key={email.email_id} className="flex justify-between">
                                                        <span className="text-sm text-muted-foreground">{email.email_id}:</span>
                                                        <span className="text-sm font-medium">
                                                            {email.centrality_score.toFixed(2)}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}
