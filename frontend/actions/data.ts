'use server';

import { db } from '@/db';
import {
	ProcessedEmail,
	processedEmails,
	summarizedEmails,
	visualizationData,
} from '@/db/schema';
import { and, count, desc, isNotNull, ne, notIlike } from 'drizzle-orm';

export async function getProcessedEmails(
	pageNum = 1,
): Promise<ProcessedEmail[]> {
	const limit = 10;
	const emails = await db
		.select()
		.from(processedEmails)
		.limit(limit)
		.offset(pageNum * limit);
	return emails;
}

export async function getTopEmailSenders() {
	const emails = await db
		.select({
			sender: summarizedEmails.from,
			count: count(),
		})
		.from(summarizedEmails)
		.where(
			and(
				isNotNull(summarizedEmails.from),
				ne(summarizedEmails.from, ''),
			),
		)
		.groupBy(summarizedEmails.from)
		.orderBy(desc(count()))
		.limit(5);
	return emails;
}

export async function getSentimentData() {
	const sentimentData = await db
		.select({
			sentiment: summarizedEmails.sentiment,
			count: count(),
		})
		.from(summarizedEmails)
		.where(
			and(
				isNotNull(summarizedEmails.sentiment),
				ne(summarizedEmails.sentiment, ''),
			),
		)
		.groupBy(summarizedEmails.sentiment);
	return sentimentData;
}

export async function getVisualizationImagesLinks() {
	const images = await db
		.select({
			name: visualizationData.fileType,
			url: visualizationData.fileUrl,
		})
		.from(visualizationData)
		.where(notIlike(visualizationData.fileUrl, '%html%'));
	return images;
}
