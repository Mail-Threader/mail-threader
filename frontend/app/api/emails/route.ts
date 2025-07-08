import { db } from '@/db';
import { processedData } from '@/db/schema';
import { inArray } from 'drizzle-orm';
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { message_ids } = await request.json();

    if (!message_ids || !Array.isArray(message_ids)) {
      return NextResponse.json({ error: 'message_ids are required' }, { status: 400 });
    }

    if (message_ids.length === 0) {
        return NextResponse.json([]);
    }

    const emails = await db
      .select({
        message_id: processedData.message_id,
        from: processedData.from,
        to: processedData.to,
        subject: processedData.subject,
        date: processedData.date,
        body: processedData.body,
      })
      .from(processedData)
      .where(inArray(processedData.message_id, message_ids));

    return NextResponse.json(emails);
  } catch (error) {
    console.error('Error fetching emails:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
