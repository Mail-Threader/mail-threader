'use client';

import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { SummarizedEmail } from '@/db/schema';
import { format } from 'date-fns';
import { Badge } from '@/components/ui/badge';

interface SummarizedEmailDetailsDialogProps {
    email: SummarizedEmail | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function SummarizedEmailDetailsDialog({
    email,
    open,
    onOpenChange,
}: SummarizedEmailDetailsDialogProps) {
    if (!email) return null;

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-3xl">
                <DialogHeader>
                    <DialogTitle>Email Summary Details</DialogTitle>
                </DialogHeader>
                <ScrollArea className="max-h-[80vh] pr-4">
                    <div className="space-y-4">
                        <div className="grid gap-2">
                            <h3 className="font-semibold">Subject</h3>
                            <p className="text-sm text-muted-foreground">{email.subject}</p>
                        </div>
                        <div className="grid gap-2">
                            <h3 className="font-semibold">From</h3>
                            <p className="text-sm text-muted-foreground">{email.from}</p>
                        </div>
                        <div className="grid gap-2">
                            <h3 className="font-semibold">To</h3>
                            <p className="text-sm text-muted-foreground">{email.to}</p>
                        </div>
                        {email.cc && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">CC</h3>
                                <p className="text-sm text-muted-foreground">{email.cc}</p>
                            </div>
                        )}
                        <div className="grid gap-2">
                            <h3 className="font-semibold">Date</h3>
                            <p className="text-sm text-muted-foreground">
                                {email.date ? format(new Date(email.date), 'PPP p') : 'N/A'}
                            </p>
                        </div>
                        <div className="grid gap-2">
                            <h3 className="font-semibold">Message ID</h3>
                            <p className="text-sm text-muted-foreground">{email.messageId}</p>
                        </div>
                        {email.originalMessageId && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">Original Message ID</h3>
                                <p className="text-sm text-muted-foreground">
                                    {email.originalMessageId}
                                </p>
                            </div>
                        )}
                        <div className="grid gap-2">
                            <h3 className="font-semibold">Sentiment</h3>
                            <Badge
                                variant={
                                    email.sentiment === 'POSITIVE'
                                        ? 'default'
                                        : email.sentiment === 'NEGATIVE'
                                            ? 'destructive'
                                            : 'secondary'
                                }
                            >
                                {email.sentiment || 'NEUTRAL'}
                            </Badge>
                        </div>
                        <div className="grid gap-2">
                            <h3 className="font-semibold">Cluster</h3>
                            <Badge variant="outline">{email.cluster || 'N/A'}</Badge>
                        </div>
                        {email.persons && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">Persons</h3>
                                <div className="flex flex-wrap gap-2">
                                    {email.persons.split(',').map((person) => (
                                        <Badge key={person} variant="secondary">
                                            {person.trim()}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}
                        {email.organizations && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">Organizations</h3>
                                <div className="flex flex-wrap gap-2">
                                    {email.organizations.split(',').map((org) => (
                                        <Badge key={org} variant="secondary">
                                            {org.trim()}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}
                        {email.locations && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">Locations</h3>
                                <div className="flex flex-wrap gap-2">
                                    {email.locations.split(',').map((location) => (
                                        <Badge key={location} variant="secondary">
                                            {location.trim()}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}
                        <div className="grid gap-2">
                            <h3 className="font-semibold">Original Body</h3>
                            <div className="rounded-md border bg-muted/50 p-4">
                                <p className="whitespace-pre-wrap text-sm text-muted-foreground">
                                    {email.body}
                                </p>
                            </div>
                        </div>
                        {email.cleanBody && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">Cleaned Body</h3>
                                <div className="rounded-md border bg-muted/50 p-4">
                                    <p className="whitespace-pre-wrap text-sm text-muted-foreground">
                                        {email.cleanBody}
                                    </p>
                                </div>
                            </div>
                        )}
                        {email.corpusSummary && (
                            <div className="grid gap-2">
                                <h3 className="font-semibold">Corpus Summary</h3>
                                <div className="rounded-md border bg-muted/50 p-4">
                                    <p className="whitespace-pre-wrap text-sm text-muted-foreground">
                                        {email.corpusSummary}
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
