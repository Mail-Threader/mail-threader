'use client';

import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { ProcessedEmail } from '@/db/schema';
import { format } from 'date-fns';

interface ProcessedEmailDetailsDialogProps {
    email: ProcessedEmail | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function ProcessedEmailDetailsDialog({
    email,
    open,
    onOpenChange,
}: ProcessedEmailDetailsDialogProps) {
    if (!email) return null;

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-3xl">
                <DialogHeader>
                    <DialogTitle>Email Details</DialogTitle>
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
                            <h3 className="font-semibold">Body</h3>
                            <div className="rounded-md border bg-muted/50 p-4">
                                <p className="whitespace-pre-wrap text-sm text-muted-foreground">
                                    {email.body}
                                </p>
                            </div>
                        </div>
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
