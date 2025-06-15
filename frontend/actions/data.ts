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
	pageSize = 10,
): Promise<{ data: ProcessedEmail[]; total: number }> {
	const [data, total] = await Promise.all([
		db
			.select()
			.from(processedEmails)
			.limit(pageSize)
			.offset((pageNum - 1) * pageSize),
		db
			.select({ count: count() })
			.from(processedEmails)
			.then((result) => result[0].count),
	]);

	return { data, total };
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

export async function getProcessedEmailsAction({
	page,
	perPage,
	sort,
	filters,
	joinOperator,
	filterFlag,
	subject,
}: {
	page: number;
	perPage: number;
	sort: { id: string; desc: boolean }[];
	filters: { id: string; value: string }[];
	joinOperator: 'and' | 'or';
	filterFlag: 'basicFilters' | 'advancedFilters';
	subject: string;
}): Promise<{ data: ProcessedEmail[]; total: number }> {
	const [data, total] = await Promise.all([
		db
			.select()
			.from(processedEmails)
			.limit(perPage)
			.offset((page - 1) * perPage),
		db
			.select({ count: count() })
			.from(processedEmails)
			.then((result) => result[0].count),
	]);

	return { data, total };
}
