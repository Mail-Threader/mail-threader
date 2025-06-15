import {
	date,
	doublePrecision,
	integer,
	json,
	pgEnum,
	pgTable,
	text,
	timestamp,
} from 'drizzle-orm/pg-core';

export const roleEnum = pgEnum('role', ['DEV', 'CLIENT']);

export const usersTable = pgTable('users', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	name: text('name'),
	email: text('email').notNull().unique(),
	role: roleEnum(),
	password: text('password').notNull(),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at')
		.defaultNow()
		.$onUpdateFn(() => new Date()),
});

export type User = typeof usersTable.$inferSelect;
export type NewUser = typeof usersTable.$inferInsert;

export const fileStorageTable = pgTable('file_storage', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	name: text('name').notNull(),
	userId: text('user_id')
		.notNull()
		.references(() => usersTable.id, {
			onDelete: 'cascade',
			onUpdate: 'cascade',
		}),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at')
		.defaultNow()
		.$onUpdateFn(() => new Date()),
	fileName: text('file_name').notNull(),
	filePath: text('file_path').notNull(),
	fileSize: text('file_size').notNull(),
	fileType: text('file_type').notNull(),
	fileUrl: text('file_url').notNull(),
	fileThumbnail: text('file_thumbnail').notNull(),
	fileThumbnailUrl: text('file_thumbnail_url').notNull(),
	fileThumbnailPath: text('file_thumbnail_path').notNull(),
	fileMetadata: json('file_metadata').notNull(),
});

export type FileStorage = typeof fileStorageTable.$inferSelect;
export type NewFileStorage = typeof fileStorageTable.$inferInsert;

export const processedEmails = pgTable('processed_emails', {
	messageId: text('message_id'),
	originalMessageId: text('original_message_id'),
	mainId: text('main_id'),
	filename: text(),
	type: text(),
	date: text(),
	from: text(),
	xFrom: text('X-From'),
	xTo: text('X-To'),
	originalSender: text('original_sender'),
	originalDate: text('original_Date'),
	to: text(),
	subject: text(),
	cc: text(),
	xCc: text('X-cc'),
	body: text(),
});

export type ProcessedEmail = typeof processedEmails.$inferSelect;

export const summarizedEmails = pgTable('summarized_emails', {
	messageId: text('message_id'),
	originalMessageId: text('original_message_id'),
	mainId: text('main_id'),
	filename: text(),
	type: text(),
	date: text(),
	from: text(),
	xFrom: text('X-From'),
	xTo: text('X-To'),
	originalSender: text('original_sender'),
	originalDate: text('original_Date'),
	to: text(),
	subject: text(),
	cc: text(),
	xCc: text('X-cc'),
	body: text(),
	cleanBody: text('clean_body'),
	tokens: text(),
	cluster: integer(),
	persons: text(),
	organizations: text(),
	locations: text(),
	sentiment: text(),
	processedDate: text('processed_date'),
	corpusSummary: text('corpus_summary'),
});

export type SummarizedEmail = typeof summarizedEmails.$inferSelect;

export const visualizationData = pgTable('visualization_data', {
	fileType: text('file_type'),
	fileUrl: text('file_url'),
});

export type VisualizationData = typeof visualizationData.$inferSelect;

export const stories = pgTable('stories', {
	title: text(),
	type: text(),
	actor: text(),
	metrics: json('metrics'),
	commonTopics: json('common_topics'),
	sampleSubjects: json('sample_subjects'),
	summary: text(),
	date: date(),
	emailCount: doublePrecision('email_count'),
	commonWords: json('common_words'),
	subject: text(),
	numEmails: doublePrecision('num_emails'),
	participants: json('participants'),
	startDate: timestamp('start_date', { mode: 'string' }),
	endDate: timestamp('end_date', { mode: 'string' }),
	topicId: text('topic_id'),
	keywords: json('keywords'),
	timestamp: text(),
	communicationPatterns: json('communication_patterns'),
	eventMetrics: json('event_metrics'),
	threadMetrics: json('thread_metrics'),
	topicMetrics: json('topic_metrics'),
	relatedEmails: json('related_emails'),
	influenceScore: doublePrecision('influence_score'),
	deviation: doublePrecision('deviation'),
	participantCount: integer('participant_count'),
	avgEmailLength: doublePrecision('avg_email_length'),
	replyRate: doublePrecision('reply_rate'),
	durationHours: doublePrecision('duration_hours'),
	avgResponseTime: doublePrecision('avg_response_time'),
	trend: text(),
	peakPeriod: text('peak_period'),
	peakCount: integer('peak_count'),
});

export type Story = typeof stories.$inferSelect;
