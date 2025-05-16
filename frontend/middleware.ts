import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { db } from './db';
import { usersTable } from './db/schema';
import { eq } from 'drizzle-orm';
import { base64url, EncryptJWT, jwtDecrypt } from 'jose';

type JWT_Token = {
	id: string;
	email: string;
};

export async function middleware(request: NextRequest) {
	const AUTH_COOKIE_NAME = 'auth_session_token';
	const cookieStore = request.cookies;
	const cookie = cookieStore.get(AUTH_COOKIE_NAME);

	if (!cookie || !cookie.value) {
		request.cookies.delete(AUTH_COOKIE_NAME);
		return NextResponse.redirect(new URL('/login', request.url));
	}

	const jwtSecret = process.env.JWT_SECRET || 'my_secret_key';

	try {
		const { payload } = await jwtDecrypt(
			cookie.value,
			base64url.decode(jwtSecret),
		);

		if (!payload || !payload.id || !payload.email) {
			request.cookies.delete(AUTH_COOKIE_NAME);
			return NextResponse.redirect(new URL('/login', request.url));
		}

		const sessionTokenObj = {
			id: payload.id,
			email: payload.email,
		} as JWT_Token;

		const existingUser = await db.query.usersTable.findFirst({
			where: eq(usersTable.email, sessionTokenObj.email),
			columns: {
				email: true,
				id: true,
			},
		});

		if (!existingUser) {
			request.cookies.delete(AUTH_COOKIE_NAME);
			return NextResponse.redirect(new URL('/login', request.url));
		}

		const jwtToken = await new EncryptJWT(sessionTokenObj)
			.setProtectedHeader({
				alg: 'dir',
				enc: 'A128CBC-HS256',
			})
			.setIssuedAt()
			.setExpirationTime('7 days')
			.encrypt(base64url.decode(jwtSecret));

		request.cookies.set(AUTH_COOKIE_NAME, jwtToken);

		return NextResponse.next();
	} catch (error) {
		return NextResponse.redirect(new URL('/login', request.url));
	}
}

// See "Matching Paths" below to learn more
export const config = {
	matcher: '/dashboard/:path*',
	runtime: 'nodejs',
};
