import { db } from '@/db';
import {
	nonThreadedStories,
	NonThreadedStory,
	threadedStories,
	ThreadedStory,
	storySummaries,
	StorySummary,
} from '@/db/schema';
import { and, desc, eq } from 'drizzle-orm';

export async function getThreadedStories(): Promise<{
	data: ThreadedStory[];
	total: number;
}> {
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

	const total = data.length;
	return { data, total };
}

export async function getNonThreadedStories(): Promise<{
	data: NonThreadedStory[];
	total: number;
}> {
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

	const total = data.length;
	return { data, total };
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
	const total = data.length;
	return { data, total };
}
