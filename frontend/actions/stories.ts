import { db } from '@/db';
import {
	nonThreadedStories,
	NonThreadedStory,
	threadedStories,
	ThreadedStory,
	storySummaries,
	StorySummary,
	processedData,
	ProcessedData,
} from '@/db/schema';
import { and, desc, eq, inArray } from 'drizzle-orm';

export async function getThreadedStories() {
	const resultType = await db
		.select({ result_type: threadedStories.resultType })
		.from(threadedStories)
		.orderBy(desc(threadedStories.createdAt))
		.limit(1);

	if (!resultType || resultType.length === 0) {
		return { data: [], total: 0 };
	}

	const data = await db
		.select()
		.from(threadedStories)
		.where(
			and(
				eq(threadedStories.resultType, resultType[0].result_type),
				eq(threadedStories.style, 'factual'),
			),
		)
		.orderBy(desc(threadedStories.createdAt));

	if (!data || data.length === 0) {
		return { data: [], total: 0 };
	}

	// fetch related emails for each story, relatedEmails is an array of message IDs to be fetched from processedData
	const messageIds = data
		.map((story) => story.relatedEmails)
		.flat() as string[];
	// select message_id, from, to, subject, date, body from processedData where message_id in (messageIds)

	const emails = await db
		.select({
			message_id: processedData.messageId,
			from: processedData.from,
			to: processedData.to,
			subject: processedData.subject,
			date: processedData.date,
			body: processedData.body,
		})
		.from(processedData)
		.where(inArray(processedData.messageId, messageIds));

	// map emails in data
	const updatedData = data.map((story) => {
		const relatedEmails = story.relatedEmails.map((messageId) => {
			return emails.find((email) => email.message_id === messageId);
		});
		return {
			...story,
			allEmails: relatedEmails.filter((email) => email !== undefined),
		};
	});

	const total = updatedData.length;
	return { data: updatedData, total };
}

export async function getNonThreadedStories() {
	const resultType = await db
		.select({ result_type: nonThreadedStories.resultType })
		.from(nonThreadedStories)
		.orderBy(desc(nonThreadedStories.createdAt))
		.limit(1);

	if (!resultType || resultType.length === 0) {
		return { data: [], total: 0 };
	}

	const data = await db
		.select()
		.from(nonThreadedStories)
		.where(
			and(
				eq(nonThreadedStories.resultType, resultType[0].result_type),
				eq(nonThreadedStories.style, 'factual'),
			),
		)
		.orderBy(desc(nonThreadedStories.createdAt));

	if (!data || data.length === 0) {
		return { data: [], total: 0 };
	}

	const messageIds = data.map((story) => story.messageId) as string[];
	// select message_id, from, to, subject, date, body from processedData where
	// message_id in (messageIds)
	const emails = await db
		.select({
			message_id: processedData.messageId,
			from: processedData.from,
			to: processedData.to,
			subject: processedData.subject,
			date: processedData.date,
			body: processedData.body,
		})
		.from(processedData)
		.where(inArray(processedData.messageId, messageIds));

	// map emails in data
	const updatedData = data.map((story) => {
		const relatedEmails = emails.filter(
			(email) => email.message_id === story.messageId,
		);
		return {
			...story,
			allEmails: relatedEmails,
		};
	});
	const total = updatedData.length;
	return { data: updatedData, total };
}

export async function getSummaries(): Promise<{
	data: StorySummary[];
	total: number;
}> {
	const resultType = await db
		.select({ result_type: storySummaries.resultType })
		.from(storySummaries)
		.orderBy(desc(storySummaries.createdAt))
		.limit(1);

	if (!resultType || resultType.length === 0) {
		return { data: [], total: 0 };
	}

	const data = await db
		.select()
		.from(storySummaries)
		.where(eq(storySummaries.resultType, resultType[0].result_type))
		.orderBy(desc(storySummaries.createdAt));
	if (!data || data.length === 0) {
		return { data: [], total: 0 };
	}

	const messageIds = data
		.map((summary) => summary.summaryMetadata.related_emails)
		.flat() as string[];
	// select message_id, from, to, subject, date, body from processedData where
	// message_id in (messageIds)

	const emails = await db
		.select({
			messageId: processedData.messageId,
			from: processedData.from,
			to: processedData.to,
			subject: processedData.subject,
			date: processedData.date,
			body: processedData.body,
		})
		.from(processedData)
		.where(inArray(processedData.messageId, messageIds));

	// map emails in data
	const updatedData = data.map((summary) => {
		const relatedEmails = summary.summaryMetadata.related_emails.map(
			(messageId) => {
				return emails.find((email) => email.messageId === messageId);
			},
		);
		return {
			...summary,
			allEmails: relatedEmails.filter((email) => email !== undefined),
		};
	});
	const total = updatedData.length;
	return { data: updatedData, total };
}
