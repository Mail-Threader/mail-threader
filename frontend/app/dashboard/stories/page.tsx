// 'use client';
import {
	getNonThreadedStories,
	getSummaries,
	getThreadedStories,
} from '@/actions/stories';

interface EmailDetail {
	message_id: string;
	from: string;
	to: string;
	subject: string;
	date: string;
	body: string;
}

async function fetchEmailDetails(messageIds: string[]): Promise<EmailDetail[]> {
	if (messageIds.length === 0) {
		return [];
	}
	try {
		const response = await fetch('/api/emails', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ message_ids: messageIds }),
		});
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		const data = await response.json();
		return data;
	} catch (error) {
		console.error('Failed to fetch email details:', error);
		return [];
	}
}

interface ThreadedStory {
	thread_id: number;
	style: string;
	title: string;
	story: string;
	related_emails: string[];
	email_count: number;
	result_type: string;
	created_at: string;
	updated_at: string;
}

interface NonThreadedStory {
	message_id: string;
	style: string;
	title: string;
	story: string;
	related_emails: string[];
	email_count: number;
	result_type: string;
	created_at: string;
	updated_at: string;
}

interface SummaryMetadata {
	metrics?: {
		sent: number;
		received: number;
		total: number;
		degree_centrality: number;
		betweenness_centrality: number;
		pagerank: number;
	};
	common_topics?: Array<[string, number]>;
	sample_subjects?: string[];
	communication_patterns?: {
		busiest_day: string;
		busiest_day_count: number;
		avg_response_time: string;
	};
	email_count?: number;
	common_words?: Array<[string, number]>;
	event_metrics?: {
		participant_count: number;
		avg_email_length: number;
		reply_rate: number;
	};
	participants?: string[];
	keywords?: string[];
	topic_metrics?: {
		trend: string;
		peak_period: string;
		peak_count: number;
	};
	related_emails?: string[];
}

interface Summary {
	summary_id: string;
	original_id: string;
	summary_title: string;
	summary_content: string;
	summary_type: 'key_actor' | 'significant_event' | 'topic_evolution';
	created_at: string;
	updated_at: string;
	summary_metadata: SummaryMetadata;
}

const fetchStories = async (): Promise<{
	threaded_stories: ThreadedStory[];
	non_threaded_stories: NonThreadedStory[];
}> => {
	// This function fetches story data directly from the database via the API.
	try {
		const response = await fetch('/api/stories');
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		const data = await response.json();
		return data;
	} catch (error) {
		console.error('Failed to fetch stories:', error);
		return { threaded_stories: [], non_threaded_stories: [] };
	}
};

const fetchSummaries = async (): Promise<Summary[]> => {
	// This function fetches summary data directly from the database via the API.
	try {
		const response = await fetch('/api/summaries');
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		const data = await response.json();
		return data.map((summary: any) => ({
			...summary,
			summary_metadata: JSON.parse(summary.summary_metadata),
		}));
	} catch (error) {
		console.error('Failed to fetch summaries:', error);
		return [];
	}
};

const EmailDetailsComponent: React.FC<{ messageIds: string[] }> = ({
	messageIds,
}) => {
	const [emails, setEmails] = useState<EmailDetail[]>([]);
	const [loading, setLoading] = useState<boolean>(true);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const getEmails = async () => {
			try {
				setLoading(true);
				const fetchedEmails = await fetchEmailDetails(messageIds);
				setEmails(fetchedEmails);
			} catch (err) {
				setError('Failed to load emails.');
				console.error(err);
			} finally {
				setLoading(false);
			}
		};
		if (messageIds && messageIds.length > 0) {
			getEmails();
		} else {
			setEmails([]);
			setLoading(false);
		}
	}, [messageIds]);

	if (loading) return <p>Loading email details...</p>;
	if (error) return <p className="text-red-500">{error}</p>;
	if (!emails || emails.length === 0) return <p>No email details found.</p>;

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
						<strong>Body Preview:</strong>{' '}
						{email.body.substring(0, 200)}...
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

	// const [threadedStories, setThreadedStories] = useState<ThreadedStory[]>([]);
	// const [nonThreadedStories, setNonThreadedStories] = useState<
	// 	NonThreadedStory[]
	// >([]);
	// const [summaries, setSummaries] = useState<Summary[]>([]);
	// const [loading, setLoading] = useState(true);
	// const [error, setError] = useState<string | null>(null);

	// useEffect(() => {
	// 	const loadData = async () => {
	// 		try {
	// 			setLoading(true);
	// 			const { threaded_stories, non_threaded_stories } =
	// 				await fetchStories();
	// 			setThreadedStories(threaded_stories);
	// 			setNonThreadedStories(non_threaded_stories);
	// 			const fetchedSummaries = await fetchSummaries();
	// 			setSummaries(fetchedSummaries);
	// 		} catch (err) {
	// 			setError('Failed to load data.');
	// 			console.error(err);
	// 		} finally {
	// 			setLoading(false);
	// 		}
	// 	};
	// 	loadData();
	// }, []);

	return <h1>Hello</h1>;

	if (loading) {
		return (
			<div className="container mx-auto p-4 text-center">
				Loading stories and summaries...
			</div>
		);
	}

	if (error) {
		return (
			<div className="container mx-auto p-4 text-center text-red-500">
				Error: {error}
			</div>
		);
	}

	return (
		<div className="container mx-auto p-4">
			<h1 className="text-3xl font-bold mb-6">Stories & Summaries</h1>

			<section className="mb-8">
				<h2 className="text-2xl font-semibold mb-4">
					Threaded Stories
				</h2>
				{threadedStories.length === 0 && (
					<p>No threaded stories found.</p>
				)}
				{threadedStories.map((story) => (
					<Accordion
						type="single"
						collapsible
						className="w-full mb-4"
						key={story.thread_id}
					>
						<AccordionItem value={`item-${story.thread_id}`}>
							<AccordionTrigger>
								<div className="flex justify-between items-center w-full pr-4">
									<span className="text-lg font-medium">
										{story.title} ({story.email_count}{' '}
										emails)
									</span>
									<Badge variant="secondary">Threaded</Badge>
								</div>
							</AccordionTrigger>
							<AccordionContent className="p-4 bg-gray-50 dark:bg-gray-900 rounded-b-md">
								<p className="mb-4">
									<strong>Narrative:</strong> {story.story}
								</p>
								<h4 className="text-lg font-semibold mb-2">
									Related Emails:
								</h4>
								<EmailDetailsComponent
									messageIds={story.related_emails}
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
				{nonThreadedStories.length === 0 && (
					<p>No non-threaded stories found.</p>
				)}
				{nonThreadedStories.map((story) => (
					<Accordion
						type="single"
						collapsible
						className="w-full mb-4"
						key={story.message_id}
					>
						<AccordionItem value={`item-${story.message_id}`}>
							<AccordionTrigger>
								<div className="flex justify-between items-center w-full pr-4">
									<span className="text-lg font-medium">
										{story.title}
									</span>
									<Badge variant="secondary">
										Non-Threaded
									</Badge>
								</div>
							</AccordionTrigger>
							<AccordionContent className="p-4 bg-gray-50 dark:bg-gray-900 rounded-b-md">
								<p className="mb-4">
									<strong>Narrative:</strong> {story.story}
								</p>
								<h4 className="text-lg font-semibold mb-2">
									Email Details:
								</h4>
								<EmailDetailsComponent
									messageIds={[story.message_id]}
								/>
							</AccordionContent>
						</AccordionItem>
					</Accordion>
				))}
			</section>

			<section className="mb-8">
				<h2 className="text-2xl font-semibold mb-4">Summaries</h2>
				{summaries.length === 0 && <p>No summaries found.</p>}
				{summaries.map((summary) => (
					<Accordion
						type="single"
						collapsible
						className="w-full mb-4"
						key={summary.summary_id}
					>
						<AccordionItem value={`item-${summary.summary_id}`}>
							<AccordionTrigger>
								<div className="flex justify-between items-center w-full pr-4">
									<span className="text-lg font-medium">
										{summary.summary_title}
									</span>
									<Badge variant="secondary">
										{summary.summary_type
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
									{summary.summary_content}
								</p>
								<p className="mb-4">
									<strong>Type:</strong>{' '}
									{summary.summary_type}
								</p>
								{summary.summary_type === 'key_actor' && (
									<div>
										<p>
											<strong>Actor:</strong>{' '}
											{summary.original_id}
										</p>
										{summary.summary_metadata.metrics && (
											<p>
												<strong>Metrics:</strong> Sent:{' '}
												{
													summary.summary_metadata
														.metrics.sent
												}
												, Received:{' '}
												{
													summary.summary_metadata
														.metrics.received
												}
												, Total:{' '}
												{
													summary.summary_metadata
														.metrics.total
												}
											</p>
										)}
										{summary.summary_metadata
											.communication_patterns && (
											<p>
												<strong>
													Communication Patterns:
												</strong>{' '}
												Busiest Day:{' '}
												{
													summary.summary_metadata
														.communication_patterns
														.busiest_day
												}{' '}
												(
												{
													summary.summary_metadata
														.communication_patterns
														.busiest_day_count
												}{' '}
												emails), Avg. Response Time:{' '}
												{
													summary.summary_metadata
														.communication_patterns
														.avg_response_time
												}
											</p>
										)}
										{summary.summary_metadata
											.common_topics && (
											<p>
												<strong>Common Topics:</strong>{' '}
												{summary.summary_metadata.common_topics
													.map((t) => t[0])
													.join(', ')}
											</p>
										)}
									</div>
								)}
								{summary.summary_type ===
									'significant_event' && (
									<div>
										<p>
											<strong>Event Date:</strong>{' '}
											{summary.original_id}
										</p>
										{summary.summary_metadata
											.email_count && (
											<p>
												<strong>Email Count:</strong>{' '}
												{
													summary.summary_metadata
														.email_count
												}
											</p>
										)}
										{summary.summary_metadata
											.common_words && (
											<p>
												<strong>Common Words:</strong>{' '}
												{summary.summary_metadata.common_words
													.map((w) => w[0])
													.join(', ')}
											</p>
										)}
										{summary.summary_metadata
											.participants && (
											<p>
												<strong>Participants:</strong>{' '}
												{summary.summary_metadata.participants.join(
													', ',
												)}
											</p>
										)}
									</div>
								)}
								{summary.summary_type === 'topic_evolution' && (
									<div>
										<p>
											<strong>Topic ID:</strong>{' '}
											{summary.original_id}
										</p>
										{summary.summary_metadata.keywords && (
											<p>
												<strong>Keywords:</strong>{' '}
												{summary.summary_metadata.keywords.join(
													', ',
												)}
											</p>
										)}
										{summary.summary_metadata
											.topic_metrics && (
											<p>
												<strong>Topic Trend:</strong>{' '}
												{
													summary.summary_metadata
														.topic_metrics.trend
												}{' '}
												(Peak:{' '}
												{
													summary.summary_metadata
														.topic_metrics
														.peak_period
												}{' '}
												with{' '}
												{
													summary.summary_metadata
														.topic_metrics
														.peak_count
												}{' '}
												emails)
											</p>
										)}
									</div>
								)}
								{summary.summary_metadata.related_emails &&
									summary.summary_metadata.related_emails
										.length > 0 && (
										<>
											<h4 className="text-lg font-semibold mb-2">
												Related Emails:
											</h4>
											<EmailDetailsComponent
												messageIds={
													summary.summary_metadata
														.related_emails
												}
											/>
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
