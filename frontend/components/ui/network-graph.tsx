import * as React from "react";
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { cn } from "@/lib/utils";

interface Node {
    id: string;
    label: string;
    value: number;
}

interface Link {
    source: string;
    target: string;
    value: number;
}

interface NetworkGraphProps extends React.HTMLAttributes<HTMLDivElement> {
    nodes: Node[];
    links: Link[];
    nodeSize?: number;
    linkWidth?: number;
    nodeColor?: (node: Node) => string;
    linkColor?: string;
    nodeLabel?: (node: Node) => string;
}

export function NetworkGraph({
    nodes,
    links,
    nodeSize = 10,
    linkWidth = 1,
    nodeColor = () => "#3b82f6",
    linkColor = "#94a3b8",
    nodeLabel,
    className,
    ...props
}: NetworkGraphProps) {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        if (!svgRef.current) return;

        const width = svgRef.current.clientWidth;
        const height = svgRef.current.clientHeight;

        // Clear previous graph
        d3.select(svgRef.current).selectAll("*").remove();

        // Create simulation
        const simulation = d3
            .forceSimulation(nodes)
            .force(
                "link",
                d3
                    .forceLink(links)
                    .id((d: any) => d.id)
                    .distance(100)
            )
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        // Create SVG
        const svg = d3.select(svgRef.current);

        // Add links
        const link = svg
            .append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke", linkColor)
            .attr("stroke-width", linkWidth);

        // Add nodes
        const node = svg
            .append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", nodeSize)
            .attr("fill", nodeColor)
            .call(drag(simulation));

        // Add labels
        const label = svg
            .append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .text((d) => (nodeLabel ? nodeLabel(d) : d.label))
            .attr("font-size", "10px")
            .attr("dx", nodeSize + 5)
            .attr("dy", 4);

        // Update positions on each tick
        simulation.on("tick", () => {
            link
                .attr("x1", (d: any) => d.source.x)
                .attr("y1", (d: any) => d.source.y)
                .attr("x2", (d: any) => d.target.x)
                .attr("y2", (d: any) => d.target.y);

            node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y);

            label.attr("x", (d: any) => d.x).attr("y", (d: any) => d.y);
        });

        // Drag behavior
        function drag(simulation: d3.Simulation<Node, undefined>) {
            function dragstarted(event: any) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event: any) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event: any) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            return d3
                .drag<SVGCircleElement, Node>()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }

        return () => {
            simulation.stop();
        };
    }, [nodes, links, nodeSize, linkWidth, nodeColor, linkColor, nodeLabel]);

    return (
        <svg
            ref={svgRef}
            className={cn("w-full h-full", className)}
            {...props}
        />
    );
}
