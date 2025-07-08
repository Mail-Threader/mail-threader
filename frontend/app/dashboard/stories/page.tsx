import {
	getNonThreadedStories,
	getSummaries,
	getThreadedStories,
} from '@/actions/stories';
import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';

const EmailDetailsComponent: React.FC<{
	emails: {
		message_id: string;
		from: string;
		to: string;
		subject: string;
		date: string;
		body: string;
	}[];
}> = ({ emails }) => {
	return (
		<div className="mt-4 space-y-4">
			{emails.map((email) => (
				<div
					key={email.message_id}
					className="border p-3 rounded-md bg-gray-50 dark:bg-gray-800"
				>
					<p>
						<strong>From:</strong> {email.from}
					</p>
					<p>
						<strong>To:</strong> {email.to}
					</p>
					<p>
						<strong>Subject:</strong> {email.subject}
					</p>
					<p>
						<strong>Date:</strong>{' '}
						{new Date(email.date).toLocaleString()}
					</p>
					<p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
						<strong>Body Preview:</strong> {email.body}...
					</p>
				</div>
			))}
		</div>
	);
};

const StoriesPage = async () => {
	const threadedStories = await getThreadedStories();
	const nonThreadedStories = await getNonThreadedStories();
	const summaries = await getSummaries();

	console.log({
		threadedStories,
		nonThreadedStories,
		summaries,
	});

	return (
		<div className="container mx-auto p-4">
			<h1 className="text-3xl font-bold mb-6">Stories & Summaries</h1>

			<section className="mb-8">
				<h2 className="text-2xl font-semibold mb-4">
					Threaded Stories
				</h2>
				{threadedStories.data.length === 0 && (
					<p>No threaded stories found.</p>
				)}
				{threadedStories.data.map((story) => (
					<Accordion
						type="single"
						collapsible
						className="w-full mb-4"
						key={story.threadId}
					>
						<AccordionItem value={`item-${story.threadId}`}>
							<AccordionTrigger>
								<div className="flex justify-between items-center w-full pr-4">
									<span className="text-xl font-bold">
										{story.title} ({story.emailCount}{' '}
										emails)
									</span>
									<Badge variant="secondary">Threaded</Badge>
								</div>
							</AccordionTrigger>
							<AccordionContent className="p-4 bg-gray-50 dark:bg-gray-900 rounded-b-md">
								<p className="mb-4 text-xl">
									<strong>Narrative:</strong> {story.story}
								</p>
								<h4 className="text-lg font-semibold mb-2">
									Related Emails:
								</h4>
								<EmailDetailsComponent
									emails={story.allEmails}
								/>
							</AccordionContent>
						</AccordionItem>
					</Accordion>
				))}
			</section>

			<section className="mb-8">
				<h2 className="text-2xl font-semibold mb-4">
					Non-Threaded Stories
				</h2>
				{nonThreadedStories.data.length === 0 && (
					<p>No non-threaded stories found.</p>
				)}
				{nonThreadedStories.data.map((story) => (
					<Accordion
						type="single"
						collapsible
						className="w-full mb-4"
						key={story.messageId}
					>
						<AccordionItem value={`item-${story.messageId}`}>
							<AccordionTrigger>
								<div className="flex justify-between items-center w-full pr-4">
									<span className="text-xl font-bold">
										{story.title}
									</span>
									<Badge variant="secondary">
										Non-Threaded
									</Badge>
								</div>
							</AccordionTrigger>
							<AccordionContent className="p-4 bg-gray-50 dark:bg-gray-900 rounded-b-md">
								<p className="mb-4 text-xl">
									<strong>Narrative:</strong> {story.story}
								</p>
								<h4 className="text-lg font-semibold mb-2">
									Email Details:
								</h4>
								<EmailDetailsComponent
									emails={[story.messageId]}
								/>
							</AccordionContent>
						</AccordionItem>
					</Accordion>
				))}
			</section>

			<section className="mb-8">
				<h2 className="text-2xl font-semibold mb-4">Summaries</h2>
				{summaries.data.length === 0 && <p>No summaries found.</p>}
				{summaries.data.map((summary) => (
					<Accordion
						type="single"
						collapsible
						className="w-full mb-4"
						key={summary.summaryId}
					>
						<AccordionItem value={`item-${summary.summaryId}`}>
							<AccordionTrigger>
								<div className="flex justify-between items-center w-full pr-4">
									<span className="text-lg font-medium">
										{summary.summaryTitle}
									</span>
									<Badge variant="secondary">
										{summary.summaryType
											.replace(/_/g, ' ')
											.replace(/\b\w/g, (l) =>
												l.toUpperCase(),
											)}
									</Badge>
								</div>
							</AccordionTrigger>
							<AccordionContent className="p-4 bg-gray-50 dark:bg-gray-900 rounded-b-md">
								<p className="mb-4">
									<strong>Summary:</strong>{' '}
									{summary.summaryContent}
								</p>
								<p className="mb-4">
									<strong>Type:</strong> {summary.summaryType}
								</p>
								{summary.summaryType === 'key_actor' && (
									<div>
										<p>
											<strong>Actor:</strong>{' '}
											{summary.originalId}
										</p>
										{summary.summaryMetadata.metrics && (
											<p>
												<strong>Metrics:</strong> Sent:{' '}
												{
													summary.summaryMetadata
														.metrics.sent
												}
												, Received:{' '}
												{
													summary.summaryMetadata
														.metrics.received
												}
												, Total:{' '}
												{
													summary.summaryMetadata
														.metrics.total
												}
											</p>
										)}
										{summary.summaryMetadata
											.communication_patterns && (
											<p>
												<strong>
													Communication Patterns:
												</strong>{' '}
												Busiest Day:{' '}
												{
													summary.summaryMetadata
														.communication_patterns
														.busiest_day
												}{' '}
												(
												{
													summary.summaryMetadata
														.communication_patterns
														.busiest_day_count
												}{' '}
												emails), Avg. Response Time:{' '}
												{
													summary.summaryMetadata
														.communication_patterns
														.avg_response_time
												}
											</p>
										)}
										{summary.summaryMetadata
											.common_topics && (
											<p>
												<strong>Common Topics:</strong>{' '}
												{summary.summaryMetadata.common_topics
													.map((t) => t[0])
													.join(', ')}
											</p>
										)}
									</div>
								)}
								{summary.summaryType ===
									'significant_event' && (
									<div>
										<p>
											<strong>Event Date:</strong>{' '}
											{summary.originalId}
										</p>
										{summary.summaryMetadata
											.email_count && (
											<p>
												<strong>Email Count:</strong>{' '}
												{
													summary.summaryMetadata
														.email_count
												}
											</p>
										)}
										{summary.summaryMetadata
											.common_words && (
											<p>
												<strong>Common Words:</strong>{' '}
												{summary.summaryMetadata.common_words
													.map((w) => w[0])
													.join(', ')}
											</p>
										)}
										{summary.summaryMetadata
											.participants && (
											<p>
												<strong>Participants:</strong>{' '}
												{summary.summaryMetadata.participants.join(
													', ',
												)}
											</p>
										)}
									</div>
								)}
								{summary.summaryType === 'topic_evolution' && (
									<div>
										<p>
											<strong>Topic ID:</strong>{' '}
											{summary.originalId}
										</p>
										{summary.summaryMetadata.keywords && (
											<p>
												<strong>Keywords:</strong>{' '}
												{summary.summaryMetadata.keywords.join(
													', ',
												)}
											</p>
										)}
										{summary.summaryMetadata
											.topic_metrics && (
											<p>
												<strong>Topic Trend:</strong>{' '}
												{
													summary.summaryMetadata
														.topic_metrics.trend
												}{' '}
												(Peak:{' '}
												{
													summary.summaryMetadata
														.topic_metrics
														.peak_period
												}{' '}
												with{' '}
												{
													summary.summaryMetadata
														.topic_metrics
														.peak_count
												}{' '}
												emails)
											</p>
										)}
									</div>
								)}
								{summary.summaryMetadata.related_emails &&
									summary.summaryMetadata.related_emails
										.length > 0 && (
										<>
											<h4 className="text-lg font-semibold mb-2">
												Related Emails:
											</h4>
											{/* <EmailDetailsComponent
												emails={
													summary.summaryMetadata
														.allEmails
												}
											/> */}
											{summary.summaryMetadata
												.allEmails &&
												summary.summaryMetadata
													.allEmails.length > 0 && (
													<EmailDetailsComponent
														emails={
															summary
																.summaryMetadata
																.allEmails
														}
													/>
												)}
										</>
									)}
							</AccordionContent>
						</AccordionItem>
					</Accordion>
				))}
			</section>
		</div>
	);
};

export default StoriesPage;
