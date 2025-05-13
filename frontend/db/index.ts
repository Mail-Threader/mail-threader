import { config } from "dotenv";
import * as schema from "./schema";
// import { drizzle } from "drizzle-orm/neon-http";

// config({ path: ".env.local" }); // or .env.local

// export const db = drizzle(process.env.DATABASE_URL!, {
// 	schema,
// });

import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

// import "dotenv/config";

config({ path: ".env.local" }); // or .env.local

const pool = new Pool({
	connectionString: process.env.DATABASE_URL,
});
export const db = drizzle(pool, {
	schema,
});
