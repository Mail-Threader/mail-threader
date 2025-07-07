import { NextResponse } from 'next/server';

export async function GET(
    request: Request,
    { params }: { params: { id: string } }
) {
    try {
        // TODO: Replace with actual database query
        // For now, return mock data
        const mockStory = {
            metadata: {
                generation_date: new Date().toISOString(),
                total_emails: 1500,
                time_span: {
                    start: '2000-01-01T00:00:00Z',
                    end: '2001-12-31T23:59:59Z',
                },
            },
            key_actors: {
                title: 'Key Actors in the Email Network',
                main_characters: [
                    {
                        name: 'Andrew Fastow',
                        role: 'CFO',
                        metrics: {
                            degree_centrality: 0.85,
                            betweenness_centrality: 0.75,
                            pagerank: 0.92,
                        },
                        influence_score: 0.84,
                    },
                    {
                        name: 'Jeff Skilling',
                        role: 'CEO',
                        metrics: {
                            degree_centrality: 0.78,
                            betweenness_centrality: 0.82,
                            pagerank: 0.88,
                        },
                        influence_score: 0.83,
                    },
                ],
                relationships: [
                    {
                        participants: ['Andrew Fastow', 'Jeff Skilling'],
                        strength: 0.9,
                        type: 'Direct Communication',
                    },
                ],
            },
            topic_evolution: {
                title: 'Evolution of Topics Over Time',
                timeline: [
                    {
                        period: 'Q1 2000',
                        topics: [
                            {
                                id: '1',
                                count: 450,
                                keywords: ['SPE', 'partnership', 'off-balance-sheet', 'debt'],
                            },
                            {
                                id: '2',
                                count: 320,
                                keywords: ['risk', 'exposure', 'hedge', 'derivatives'],
                            },
                        ],
                    },
                ],
                key_topics: [
                    {
                        id: '1',
                        keywords: ['SPE', 'partnership', 'off-balance-sheet', 'debt'],
                        significance: 0.9,
                    },
                ],
            },
            significant_events: {
                title: 'Significant Events Timeline',
                events: [
                    {
                        date: '2000-03-15T00:00:00Z',
                        significance: 0.9,
                        description: 'LJM2 partnership formation',
                        key_topics: ['SPE', 'partnership'],
                        sample_subjects: [
                            'LJM2 Partnership Structure',
                            'Re: LJM2 Partnership Structure',
                        ],
                    },
                ],
            },
            email_threads: {
                title: 'Major Email Threads',
                threads: [
                    {
                        subject: 'LJM2 Partnership Structure',
                        size: 45,
                        participants: [
                            'andrew.fastow@enron.com',
                            'jeff.skilling@enron.com',
                        ],
                        timeline: {
                            start: '2000-03-15T00:00:00Z',
                            end: '2000-03-16T23:59:59Z',
                        },
                    },
                ],
            },
            email_connections: {
                title: 'Email Network Analysis',
                network_properties: {
                    total_connections: 2500,
                    connection_types: {
                        direct: 1500,
                        cc: 800,
                        bcc: 200,
                    },
                    communities: 12,
                },
                central_emails: [
                    {
                        email_id: 'email_001',
                        centrality_score: 0.95,
                    },
                    {
                        email_id: 'email_002',
                        centrality_score: 0.88,
                    },
                ],
                temporal_patterns: {
                    peak_hours: [9, 14, 16],
                    peak_days: ['Tuesday', 'Wednesday', 'Thursday'],
                },
            },
            narrative_summary:
                'This analysis reveals the complex network of relationships and communications within Enron, focusing on key actors like Andrew Fastow and Jeff Skilling. The story uncovers patterns of communication, significant events, and the evolution of topics over time, particularly around financial engineering and risk management.',
        };

        return NextResponse.json(mockStory);
    } catch (error) {
        console.error('Error fetching story:', error);
        return NextResponse.json(
            { error: 'Failed to fetch story' },
            { status: 500 }
        );
    }
}
