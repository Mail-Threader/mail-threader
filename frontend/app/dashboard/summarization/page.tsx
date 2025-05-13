
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileTextIcon, ZapIcon } from "lucide-react";
import Image from "next/image";

interface SummaryItem {
  id: string;
  title: string;
  content: string;
  keywords: string[];
  dateGenerated: string;
  sourceEmails: number;
}

const placeholderSummaries: SummaryItem[] = [
  { id: "SUM001", title: "Q4 Financial Performance Overview", content: "The fourth quarter showed strong revenue growth driven by new energy contracts. However, operating expenses also increased, impacting overall profitability. Key concerns revolve around derivative accounting practices.", keywords: ["finance", "revenue", "Q4", "profitability"], dateGenerated: "2001-01-20", sourceEmails: 152 },
  { id: "SUM002", title: "Project Raptor: Risk Assessment", content: "Project Raptor involves complex special purpose entities (SPEs) designed to hedge risk and manage debt. Analysis indicates potential accounting irregularities and off-balance-sheet liabilities that could pose significant financial risk.", keywords: ["Project Raptor", "SPEs", "risk", "accounting"], dateGenerated: "2001-03-05", sourceEmails: 88 },
  { id: "SUM003", title: "California Energy Crisis Communications", content: "Internal communications reveal discussions about strategies to manage public perception and regulatory scrutiny during the California energy crisis. Topics include price manipulation allegations and Enron's role in market dynamics.", keywords: ["California", "energy crisis", "regulation", "PR"], dateGenerated: "2001-05-15", sourceEmails: 230 },
];

export default function SummarizationPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Email Summaries</CardTitle>
          <CardDescription>
            Explore AI-generated summaries focusing on pertinent details from the Enron email dataset.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            {placeholderSummaries.map((summary, index) => (
              <AccordionItem value={`item-${index}`} key={summary.id}>
                <AccordionTrigger>
                  <div className="flex items-center gap-3">
                    <FileTextIcon className="h-5 w-5 text-primary" />
                    <span className="font-medium">{summary.title}</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-3 pl-8">
                  <p className="text-muted-foreground">{summary.content}</p>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p><strong>Date Generated:</strong> {summary.dateGenerated}</p>
                    <p><strong>Source Emails Analyzed:</strong> {summary.sourceEmails}</p>
                    <div className="flex items-center gap-2 pt-1">
                      <strong>Keywords:</strong>
                      {summary.keywords.map(keyword => (
                        <Badge variant="secondary" key={keyword}>{keyword}</Badge>
                      ))}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </CardContent>
      </Card>

       <Card>
        <CardHeader>
          <CardTitle>Summarization Insights</CardTitle>
          <CardDescription>Key themes and topics emerging from summarization.</CardDescription>
        </CardHeader>
        <CardContent className="grid md:grid-cols-3 gap-4">
          {[
            { title: "Financial Irregularities", hint: "financial chart", icon: ZapIcon },
            { title: "Regulatory Concerns", hint: "legal document", icon: ZapIcon },
            { title: "Market Manipulation", hint: "stock market", icon: ZapIcon },
          ].map((insight, index) => (
            <Card key={insight.title} className="shadow-md hover:shadow-lg transition-shadow">
              <CardContent className="p-4 flex flex-col items-center text-center">
                <Image
                  src={`https://picsum.photos/300/200?random=${10 + index}`}
                  alt={insight.title}
                  width={300}
                  height={200}
                  className="rounded-md mb-3 w-full h-auto object-cover"
                  data-ai-hint={insight.hint}
                />
                <insight.icon className="h-8 w-8 text-primary mb-2" />
                <h3 className="font-semibold text-md">{insight.title}</h3>
              </CardContent>
            </Card>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
