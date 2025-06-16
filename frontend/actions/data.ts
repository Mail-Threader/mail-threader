'use server';

import { db } from '@/db';
import {
	ProcessedEmail,
	processedEmails,
	summarizedEmails,
	visualizationData,
} from '@/db/schema';
import { and, count, desc, isNotNull, ne, notIlike, asc, sql } from 'drizzle-orm';

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

export async function getProcessedEmailDateRange() {
	// Get min and max date from processedEmails
	const [minResult] = await db.select({ min: sql`MIN(date)` }).from(processedEmails);
	const [maxResult] = await db.select({ max: sql`MAX(date)` }).from(processedEmails);
	return {
		min: minResult?.min ?? null,
		max: maxResult?.max ?? null,
	};
}

export async function getProcessedEmailsAction({
	page,
	perPage,
	sort,
	filters,
	joinOperator,
	filterFlag,
	subject,
	dateRange,
}: {
	page: number;
	perPage: number;
	sort: { id: string; desc: boolean }[];
	filters: { id: string; value: string }[];
	joinOperator: 'and' | 'or';
	filterFlag: 'basicFilters' | 'advancedFilters';
	subject: string;
	dateRange?: { from: string; to: string };
}): Promise<{ data: ProcessedEmail[]; total: number }> {
	// Build where clause
	let whereClauses = [];
	for (const filter of filters) {
		if (filter.id === 'date' && filter.value) {
			// Expecting value to be a stringified array: [from, to]
			try {
				const [from, to] = JSON.parse(filter.value);
				if (from && to) {
					whereClauses.push(sql`date >= ${from} AND date <= ${to}`);
				} else if (from) {
					whereClauses.push(sql`date >= ${from}`);
				} else if (to) {
					whereClauses.push(sql`date <= ${to}`);
				}
			} catch { }
		} else if (filter.value) {
			whereClauses.push(sql`${sql.raw(filter.id)} ILIKE '%' || ${filter.value} || '%'`);
		}
	}
	if (subject) {
		whereClauses.push(sql`subject ILIKE '%' || ${subject} || '%'`);
	}
	const where = whereClauses.length > 0 ? (joinOperator === 'and' ? sql`${whereClauses.join(' AND ')}` : sql`${whereClauses.join(' OR ')}`) : undefined;

	// Sorting
	let orderBy = undefined;
	if (sort && sort.length > 0) {
		orderBy = sort.map((s) => (s.desc ? desc(sql.raw(s.id)) : asc(sql.raw(s.id))));
	}

	// Build query for data
	let dataQuery = db.select().from(processedEmails);
	let countQuery = db.select({ count: count() }).from(processedEmails);
	if (where) {
		dataQuery = dataQuery.where(where);
		countQuery = countQuery.where(where);
	}
	if (orderBy && Array.isArray(orderBy) && orderBy.length > 0) {
		dataQuery = dataQuery.orderBy(...orderBy);
	}
	dataQuery = dataQuery.limit(perPage).offset((page - 1) * perPage);

	const [data, totalResult] = await Promise.all([
		dataQuery.execute(),
		countQuery.execute(),
	]);
	const total = totalResult[0]?.count ?? 0;

	return { data, total };
}
