import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { NetworkGraph } from "@/components/ui/network-graph";

interface StoryNetworkProps {
    network: {
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
}

export function StoryNetwork({ network }: StoryNetworkProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-lg">Communication Network</CardTitle>
                <p className="text-sm text-muted-foreground">
                    Visualization of email communication patterns
                </p>
            </CardHeader>
            <CardContent>
                <div className="h-[300px] w-full">
                    <NetworkGraph
                        nodes={network.nodes}
                        links={network.links}
                        nodeSize={20}
                        linkWidth={2}
                        nodeColor={(node) => {
                            const value = node.value;
                            if (value > 0.7) return "#ef4444"; // red
                            if (value > 0.4) return "#f59e0b"; // yellow
                            return "#3b82f6"; // blue
                        }}
                        linkColor="#94a3b8"
                        nodeLabel={(node) => `${node.label} (${(node.value * 100).toFixed(0)}%)`}
                    />
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4">
                    <div className="flex items-center space-x-2">
                        <div className="h-3 w-3 rounded-full bg-red-500" />
                        <span className="text-sm text-muted-foreground">High Influence</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="h-3 w-3 rounded-full bg-yellow-500" />
                        <span className="text-sm text-muted-foreground">Medium Influence</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="h-3 w-3 rounded-full bg-blue-500" />
                        <span className="text-sm text-muted-foreground">Low Influence</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
