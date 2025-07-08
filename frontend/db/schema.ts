import {
	integer,
	json,
	jsonb,
	pgEnum,
	pgTable,
	text,
	timestamp,
	varchar,
	numeric,
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

// Tables based on src/database/database_schema_config.sql

export const processedData = pgTable('processed_emails', {
	messageId: varchar('message_id', { length: 255 }).primaryKey(),
	mainId: varchar('main_id', { length: 255 }),
	filename: varchar('filename', { length: 255 }),
	type: varchar('type', { length: 50 }),
	date: timestamp('date'),
	from: varchar('from', { length: 255 }),
	to: varchar('to', { length: 255 }),
	subject: text('subject'),
	body: text('body'),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at').defaultNow(),
	resultType: varchar('result_type', { length: 50 }).notNull(),
});

export type ProcessedData = typeof processedData.$inferSelect;

export const analysisResult = pgTable('analysis_result', {
	messageId: varchar('message_id', { length: 255 }).primaryKey(),
	mainId: varchar('main_id', { length: 255 }),
	filename: varchar('filename', { length: 255 }),
	type: varchar('type', { length: 50 }),
	date: timestamp('date'),
	from: varchar('from', { length: 255 }),
	to: varchar('to', { length: 255 }),
	subject: text('subject'),
	body: text('body'),
	entities: jsonb('entities'),
	sentiment: jsonb('sentiment'),
	dominantTopic: integer('dominant_topic'),
	topicStrength: numeric('topic_strength'),
	topicLabel: varchar('topic_label', { length: 255 }),
	cluster: integer('cluster'),
	threadId: integer('thread_id'),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at').defaultNow(),
	resultType: varchar('result_type', { length: 50 }).notNull(),
});

export type AnalysisResult = typeof analysisResult.$inferSelect;

export const threadedStories = pgTable('threaded_stories', {
	threadId: integer('thread_id').notNull(),
	style: varchar('style', { length: 50 }).notNull(),
	title: text('title'),
	story: text('story'),
	relatedEmails: jsonb('related_emails'),
	emailCount: integer('email_count'),
	resultType: varchar('result_type', { length: 50 }).notNull(),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at').defaultNow(),
});

export type ThreadedStory = typeof threadedStories.$inferSelect;

export const nonThreadedStories = pgTable('non_threaded_stories', {
	messageId: varchar('message_id', { length: 255 }).notNull(),
	style: varchar('style', { length: 50 }).notNull(),
	title: text('title'),
	story: text('story'),
	relatedEmails: jsonb('related_emails'),
	emailCount: integer('email_count'),
	resultType: varchar('result_type', { length: 50 }).notNull(),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at').defaultNow(),
});

export type NonThreadedStory = typeof nonThreadedStories.$inferSelect;

export const storySummaries = pgTable('summaries', {
	summaryId: varchar('summary_id', { length: 255 }).primaryKey(),
	originalId: varchar('original_id', { length: 255 }),
	summaryTitle: text('summary_title'),
	summaryContent: text('summary_content'),
	summaryType: varchar('summary_type', { length: 50 }),
	createdAt: timestamp('created_at').defaultNow(),
	updatedAt: timestamp('updated_at').defaultNow(),
	summaryMetadata: jsonb('summary_metadata'),
	resultType: varchar('result_type', { length: 50 }).notNull(),
});

export type StorySummary = typeof storySummaries.$inferSelect;

// export const topicAnalysis = pgTable('topic_analysis', {
// 	id: serial('id').primaryKey(),
// 	topicName: varchar('topic_name', { length: 255 }).notNull(),
// 	keywords: jsonb('keywords'),
// 	topicDistribution: jsonb('topic_distribution'),
// 	createdAt: timestamp('created_at').defaultNow(),
// 	resultType: varchar('result_type', { length: 50 }).notNull(),
// });

// export type TopicAnalysis = typeof topicAnalysis.$inferSelect;

// export const clusterAnalysis = pgTable('cluster_analysis', {
// 	id: serial('id').primaryKey(),
// 	clusterName: varchar('cluster_name', { length: 255 }).notNull(),
// 	clusterSize: integer('cluster_size'),
// 	commonWords: jsonb('common_words'),
// 	clusterMetrics: jsonb('cluster_metrics'),
// 	createdAt: timestamp('created_at').defaultNow(),
// 	resultType: varchar('result_type', { length: 50 }).notNull(),
// });

// export type ClusterAnalysis = typeof clusterAnalysis.$inferSelect;

// export const entityAnalysis = pgTable(
// 	'entity_analysis',
// 	{
// 		id: serial('id').primaryKey(),
// 		entityType: varchar('entity_type', { length: 50 }).notNull(),
// 		entityName: varchar('entity_name', { length: 255 }).notNull(),
// 		frequency: integer('frequency'),
// 		context: jsonb('context'),
// 		createdAt: timestamp('created_at').defaultNow(),
// 		resultType: varchar('result_type', { length: 50 }).notNull(),
// 	},
// 	(table) => {
// 		return {
// 			uniqueEntityTypeEntityName: unique(
// 				'idx_entity_analysis_type_name',
// 			).on(table.entityType, table.entityName),
// 		};
// 	},
// );

// export type EntityAnalysis = typeof entityAnalysis.$inferSelect;

// export const sentimentAnalysis = pgTable(
// 	'sentiment_analysis',
// 	{
// 		id: serial('id').primaryKey(),
// 		emailId: integer('email_id').references(() => processedData.id),
// 		sentimentType: varchar('sentiment_type', { length: 50 }),
// 		sentimentScore: doublePrecision('sentiment_score'),
// 		confidenceScore: doublePrecision('confidence_score'),
// 		createdAt: timestamp('created_at').defaultNow(),
// 		resultType: varchar('result_type', { length: 50 }).notNull(),
// 	},
// 	(table) => {
// 		return {
// 			emailIdIdx: index('idx_sentiment_analysis_email_id').on(
// 				table.emailId,
// 			),
// 		};
// 	},
// );

// export type SentimentAnalysis = typeof sentimentAnalysis.$inferSelect;

// export const summarizationResults = pgTable(
// 	'summarization_results',
// 	{
// 		id: serial('id').primaryKey(),
// 		emailId: integer('email_id').references(() => processedData.id),
// 		summaryStyle: varchar('summary_style', { length: 50 }),
// 		summaryText: text('summary_text'),
// 		wordCount: integer('word_count'),
// 		sentenceCount: integer('sentence_count'),
// 		entityCount: integer('entity_count'),
// 		actionItemCount: integer('action_item_count'),
// 		keyInformation: jsonb('key_information'),
// 		createdAt: timestamp('created_at').defaultNow(),
// 		resultType: varchar('result_type', { length: 50 }).notNull(),
// 	},
// 	(table) => {
// 		return {
// 			emailIdIdx: index('idx_summarization_results_email_id').on(
// 				table.emailId,
// 			),
// 		};
// 	},
// );

// export type SummarizationResult = typeof summarizationResults.$inferSelect;

// export const visualizationData = pgTable('visualization_data', {
// 	id: serial('id').primaryKey(),
// 	visualizationType: varchar('visualization_type', { length: 50 }).notNull(),
// 	fileUrl: text('file_url'),
// 	metadata: jsonb('metadata'),
// 	createdAt: timestamp('created_at').defaultNow(),
// 	resultType: varchar('result_type', { length: 50 }).notNull(),
// });

// export type VisualizationData = typeof visualizationData.$inferSelect;

// export const analysisMetadata = pgTable('analysis_metadata', {
// 	id: serial('id').primaryKey(),
// 	analysisType: varchar('analysis_type', { length: 50 }).notNull(),
// 	parameters: jsonb('parameters'),
// 	startTime: timestamp('start_time'),
// 	endTime: timestamp('end_time'),
// 	status: varchar('status', { length: 50 }),
// 	createdAt: timestamp('created_at').defaultNow(),
// 	resultType: varchar('result_type', { length: 50 }).notNull(),
// });

// export type AnalysisMetadata = typeof analysisMetadata.$inferSelect;

// export const stories = pgTable(
// 	'stories',
// 	{
// 		id: serial('id').primaryKey(),
// 		threadId: integer('thread_id'),
// 		title: text('title'),
// 		story: text('story'),
// 		relatedEmails: jsonb('related_emails'),
// 		emailCount: integer('email_count'),
// 		messageId: varchar('message_id', { length: 255 }),
// 		createdAt: timestamp('created_at').defaultNow(),
// 		resultType: varchar('result_type', { length: 50 }).notNull(),
// 	},
// 	(table) => {
// 		return {
// 			threadIdIdx: index('idx_stories_thread_id').on(table.threadId),
// 		};
// 	},
// );

// export type Story = typeof stories.$inferSelect;
