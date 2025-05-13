import { sql } from "drizzle-orm";
import { integer, pgTable, text } from "drizzle-orm/pg-core";

export const usersTable = pgTable("users", {
	id: text("id")
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	email: text("email").notNull().unique(),
	hashedPassword: text("hashed_password").notNull(),
	createdAt: integer("created_at").default(sql`(strftime('%s', 'now'))`),
	updatedAt: integer("updated_at")
		.default(sql`(strftime('%s', 'now'))`)
		.$onUpdate(() => Date.now()),
});

export type User = typeof usersTable.$inferSelect;
export type NewUser = typeof usersTable.$inferInsert;
