import { json, pgEnum, pgTable, text, timestamp } from 'drizzle-orm/pg-core';

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
