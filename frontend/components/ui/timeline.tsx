import * as React from "react";
import { cn } from "@/lib/utils";

const Timeline = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn("space-y-4", className)}
        {...props}
    />
));
Timeline.displayName = "Timeline";

const TimelineItem = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn("pl-4 pb-4", className)}
        {...props}
    />
));
TimelineItem.displayName = "TimelineItem";

export { Timeline, TimelineItem };
