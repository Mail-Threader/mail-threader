
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { LightbulbIcon, Share2Icon, ThumbsUpIcon } from "lucide-react";
import Image from "next/image";

interface Story {
  id: string;
  title: string;
  narrative: string;
  keyInsights: string[];
  relatedVisualizations: string[]; // IDs or links to visualizations
  dateDiscovered: string;
  tags: string[];
}

const placeholderStories: Story[] = [
  {
    id: "STY001",
    title: "The Rise and Fall of LJM: A Tale of Off-Balance-Sheet Deception",
    narrative: "This story uncovers the intricate web of Special Purpose Entities (SPEs), particularly LJM, created by Andrew Fastow. Emails reveal how these entities were used to hide debt, inflate earnings, and enrich executives, ultimately contributing to Enron's collapse.",
    keyInsights: ["Misleading financial reporting through SPEs.", "Conflicts of interest for executives involved in LJM.", "Lack of transparency and oversight."],
    relatedVisualizations: ["Email Volume by Sender (Fastow)", "Financial Irregularities Timeline"],
    dateDiscovered: "2001-10-25",
    tags: ["LJM", "SPE", "Fastow", "Deception", "Finance"]
  },
  {
    id: "STY002",
    title: "California's Power Play: Enron's Role in the Energy Crisis",
    narrative: "Emails suggest Enron traders exploited California's deregulated energy market, contributing to rolling blackouts and price spikes. This story explores the strategies discussed internally, including 'Death Star' and 'Get Shorty', and the subsequent public and regulatory fallout.",
    keyInsights: ["Market manipulation tactics.", "Impact on California consumers.", "Regulatory loopholes exploited."],
    relatedVisualizations: ["Energy Price Spike Chart", "Trader Communication Network"],
    dateDiscovered: "2001-04-10",
    tags: ["California", "Energy Crisis", "Market Manipulation", "Regulation"]
  },
  {
    id: "STY003",
    title: "The Culture of Secrecy: How Internal Communication Masked Problems",
    narrative: "An analysis of communication patterns reveals a culture where bad news was suppressed and overly optimistic projections were favored. This story highlights how a lack of candid internal discussion allowed critical issues to fester.",
    keyInsights: ["Suppression of negative information.", "Overemphasis on positive spin.", "Breakdown of internal controls."],
    relatedVisualizations: ["Sentiment Analysis Over Time", "Keyword Frequency (Optimism vs. Concern)"],
    dateDiscovered: "2001-11-15",
    tags: ["Corporate Culture", "Communication", "Secrecy", "Internal Controls"]
  }
];

export default function StoryExplorerPage() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Generated Stories & Insights</CardTitle>
          <CardDescription>
            Discover compelling narratives and key insights derived from the Enron email dataset.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {placeholderStories.map((story) => (
              <Card key={story.id} className="flex flex-col">
                <CardHeader>
                  <div className="mb-2">
                    <Image
                      src={`https://picsum.photos/400/200?random=${story.id.replace(/\D/g, '')}`}
                      alt={story.title}
                      width={400}
                      height={200}
                      className="rounded-md w-full h-auto object-cover"
                      data-ai-hint={`${story.tags[0]} ${story.tags[1] || 'abstract'}`}
                    />
                  </div>
                  <CardTitle className="text-lg">{story.title}</CardTitle>
                  <CardDescription>Discovered: {story.dateDiscovered}</CardDescription>
                </CardHeader>
                <CardContent className="flex-grow">
                  <p className="text-sm text-muted-foreground mb-3 line-clamp-3">{story.narrative}</p>
                  <Accordion type="single" collapsible className="w-full text-sm">
                    <AccordionItem value="insights">
                      <AccordionTrigger className="py-2 text-primary hover:no-underline">
                        <div className="flex items-center gap-2">
                         <LightbulbIcon className="h-4 w-4" /> Key Insights
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="pt-2 pl-6">
                        <ul className="list-disc space-y-1 text-muted-foreground">
                          {story.keyInsights.map((insight, idx) => (
                            <li key={idx}>{insight}</li>
                          ))}
                        </ul>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardContent>
                <CardFooter className="flex justify-between items-center pt-4 border-t">
                  <div className="flex gap-1">
                    {story.tags.slice(0, 2).map(tag => (
                      <Badge variant="outline" key={tag}>{tag}</Badge>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Button variant="ghost" size="icon" aria-label="Like story">
                      <ThumbsUpIcon className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" aria-label="Share story">
                      <Share2Icon className="h-4 w-4" />
                    </Button>
                  </div>
                </CardFooter>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
