CREATE TYPE "public"."role" AS ENUM ('DEV', 'CLIENT');

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS analysis_result (
		"message_id" varchar(255) PRIMARY KEY NOT NULL,
		"main_id" varchar(255),
		"filename" varchar(255),
		"type" varchar(50),
		"date" timestamp,
		"from" varchar(255),
		"to" varchar(255),
		"subject" text,
		"body" text,
		"entities" jsonb,
		"sentiment" jsonb,
		"dominant_topic" integer,
		"topic_strength" numeric,
		"topic_label" varchar(255),
		"cluster" integer,
		"thread_id" integer,
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now (),
		"result_type" varchar(50) NOT NULL
	);

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS file_storage (
		"id" text PRIMARY KEY NOT NULL,
		"name" text NOT NULL,
		"user_id" text NOT NULL,
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now (),
		"file_name" text NOT NULL,
		"file_path" text NOT NULL,
		"file_size" text NOT NULL,
		"file_type" text NOT NULL,
		"file_url" text NOT NULL,
		"file_thumbnail" text NOT NULL,
		"file_thumbnail_url" text NOT NULL,
		"file_thumbnail_path" text NOT NULL,
		"file_metadata" json NOT NULL
	);

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS summaries (
		"summary_id" varchar(255) PRIMARY KEY NOT NULL,
		"original_id" varchar(255),
		"summary_title" text,
		"summary_content" text,
		"summary_type" varchar(50),
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now (),
		"summary_metadata" jsonb,
		"result_type" varchar(50) NOT NULL
	);

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS non_threaded_stories (
		"message_id" varchar(255) NOT NULL,
		"style" varchar(50) NOT NULL,
		"title" text,
		"story" text,
		"related_emails" jsonb,
		"email_count" integer,
		"result_type" varchar(50) NOT NULL,
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now ()
	);

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS processed_data (
		"message_id" varchar(255) PRIMARY KEY NOT NULL,
		"main_id" varchar(255),
		"filename" varchar(255),
		"type" varchar(50),
		"date" timestamp,
		"from" varchar(255),
		"to" varchar(255),
		"subject" text,
		"body" text,
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now (),
		"result_type" varchar(50) NOT NULL
	);

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS threaded_stories (
		"thread_id" integer NOT NULL,
		"style" varchar(50) NOT NULL,
		"title" text,
		"story" text,
		"related_emails" jsonb,
		"email_count" integer,
		"result_type" varchar(50) NOT NULL,
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now ()
	);

--> statement-breakpoint
CREATE TABLE
	IF NOT EXISTS users (
		"id" text PRIMARY KEY NOT NULL,
		"name" text,
		"email" text NOT NULL,
		"role" "role",
		"password" text NOT NULL,
		"created_at" timestamp DEFAULT now (),
		"updated_at" timestamp DEFAULT now (),
		CONSTRAINT "users_email_unique" UNIQUE ("email")
	);
