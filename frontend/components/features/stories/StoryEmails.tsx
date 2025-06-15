import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/ui/accordion";

interface StoryEmailsProps {
    emails: Array<{
        subject: string;
        date: string;
        from: string;
        to: string;
        bodyPreview: string;
        category: string;
    }>;
}

export function StoryEmails({ emails }: StoryEmailsProps) {
    // Group emails by category
    const groupedEmails = emails.reduce((acc, email) => {
        if (!acc[email.category]) {
            acc[email.category] = [];
        }
        acc[email.category].push(email);
        return acc;
    }, {} as Record<string, typeof emails>);

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-lg">Related Emails</CardTitle>
                <p className="text-sm text-muted-foreground">
                    Key communications related to this story
                </p>
            </CardHeader>
            <CardContent>
                <Accordion type="multiple" className="w-full">
                    {Object.entries(groupedEmails).map(([category, categoryEmails]) => (
                        <AccordionItem key={category} value={category}>
                            <AccordionTrigger className="text-sm font-medium">
                                {category} ({categoryEmails.length})
                            </AccordionTrigger>
                            <AccordionContent>
                                <div className="space-y-4">
                                    {categoryEmails.map((email, index) => (
                                        <div
                                            key={index}
                                            className="border rounded-lg p-4 space-y-2"
                                        >
                                            <div className="flex items-center justify-between">
                                                <div className="font-medium">{email.subject}</div>
                                                <Badge variant="outline">
                                                    {format(new Date(email.date), "MMM d, yyyy")}
                                                </Badge>
                                            </div>
                                            <div className="text-sm text-muted-foreground">
                                                From: {email.from}
                                            </div>
                                            <div className="text-sm text-muted-foreground">
                                                To: {email.to}
                                            </div>
                                            <p className="text-sm text-muted-foreground mt-2">
                                                {email.bodyPreview}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </AccordionContent>
                        </AccordionItem>
                    ))}
                </Accordion>
            </CardContent>
        </Card>
    );
}
