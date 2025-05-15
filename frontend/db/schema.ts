import { pgEnum, pgTable, text, timestamp } from 'drizzle-orm/pg-core';

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
