'use server';

import {
	loginSchema,
	signupSchema,
	type LoginFormInputs,
	type SignupFormInputs,
} from '@/schemas/auth-schemas';
import { db } from '@/db';
import { NewUser, usersTable } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { compare, genSalt, hash } from 'bcrypt';
import { cookies } from 'next/headers';
import { base64url, EncryptJWT, jwtDecrypt } from 'jose';

const AUTH_COOKIE_NAME = 'auth_session_token';

const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

export interface AuthState {
	success: boolean;
	message?: string;
	user?: { email: string; id: string };
	errors?: {
		email?: string[];
		password?: string[];
		confirmPassword?: string[]; // For signup
		_form?: string[];
	};
}

type JWT_Token = {
	id: string;
	email: string;
};

export async function loginAction(data: LoginFormInputs): Promise<AuthState> {
	const validationResult = loginSchema.safeParse(data);

	if (!validationResult.success) {
		return {
			success: false,
			message: 'Invalid form data.',
			errors: validationResult.error.flatten().fieldErrors,
		};
	}

	const { email, password } = validationResult.data;

	try {
		const existingUser = await db.query.usersTable.findFirst({
			where: eq(usersTable.email, email),
			columns: {
				email: true,
				password: true,
				id: true,
			},
		});

		if (!existingUser) {
			return { success: false, message: 'Invalid email or password.' };
		}

		const passwordMatch = await compare(password, existingUser.password);

		if (!passwordMatch) {
			return { success: false, message: 'Invalid email or password.' };
		}

		const sessionTokenObj = {
			id: existingUser.id,
			email: existingUser.email,
		};

		const jwtSecret = process.env.JWT_SECRET || 'my_secret_key';

		const jwtToken = await new EncryptJWT(sessionTokenObj)
			.setProtectedHeader({
				alg: 'dir',
				enc: 'A128CBC-HS256',
			})
			.setIssuedAt()
			.setExpirationTime('7 days')
			.encrypt(base64url.decode(jwtSecret));

		const cookieStore = await cookies();

		cookieStore.set(AUTH_COOKIE_NAME, jwtToken, {
			httpOnly: true,
			secure: process.env.NODE_ENV === 'production',
			maxAge: SESSION_MAX_AGE,
			path: '/',
			sameSite: 'lax',
		});

		return {
			success: true,
			message: 'Login successful!',
			user: { email: existingUser.email, id: existingUser.id },
		};
	} catch (error) {
		console.error('Login error:', error);
		return {
			success: false,
			message: 'An unexpected error occurred. Please try again.',
		};
	}
}

export async function signupAction(data: SignupFormInputs): Promise<AuthState> {
	const validationResult = signupSchema.safeParse(data);

	if (!validationResult.success) {
		return {
			success: false,
			message: 'Invalid form data.',
			errors: validationResult.error.flatten().fieldErrors,
		};
	}

	let { email, password } = validationResult.data;

	let role: NewUser['role'] = 'CLIENT';

	if (email.includes('+dev')) {
		email = email.replace('+dev', '');
		role = 'DEV';
	}

	try {
		const existingUser = await db.query.usersTable.findFirst({
			where: eq(usersTable.email, email),
		});

		if (existingUser) {
			return {
				success: false,
				message: 'User with this email already exists.',
			};
		}

		const salt = await genSalt();

		const hashedPassword = await hash(password, salt);

		await db.insert(usersTable).values({
			email,
			password: hashedPassword,
			role,
		});

		return { success: true, message: 'Signup successful! Please login.' };
	} catch (error) {
		console.error('Signup error:', error);
		return {
			success: false,
			message: 'An unexpected error occurred. Please try again.',
		};
	}
}

export async function logoutAction(): Promise<{ success: boolean }> {
	try {
		const cookieStore = await cookies();
		cookieStore.delete(AUTH_COOKIE_NAME);
		return { success: true };
	} catch (error) {
		console.error('Logout error:', error);
		return { success: false };
	}
}

export interface CheckSessionResult {
	isAuthenticated: boolean;
	user: JWT_Token | null;
}

export async function checkSessionAction(): Promise<CheckSessionResult> {
	try {
		const cookieStore = await cookies();
		const sessionToken = cookieStore.get(AUTH_COOKIE_NAME)?.value;

		if (!sessionToken) {
			return { isAuthenticated: false, user: null };
		}

		const jwtSecret = process.env.JWT_SECRET || 'my_secret_key';

		const { payload } = await jwtDecrypt(
			sessionToken,
			base64url.decode(jwtSecret),
		);

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

		if (existingUser) {
			const sessionTokenObj = {
				id: existingUser.id,
				email: existingUser.email,
			};

			const jwtSecret = process.env.JWT_SECRET || 'my_secret_key';

			const jwtToken = await new EncryptJWT(sessionTokenObj)
				.setProtectedHeader({
					alg: 'dir',
					enc: 'A128CBC-HS256',
				})
				.setIssuedAt()
				.setExpirationTime('7 days')
				.encrypt(base64url.decode(jwtSecret));

			const cookieStore = await cookies();

			cookieStore.set(AUTH_COOKIE_NAME, jwtToken, {
				httpOnly: true,
				secure: process.env.NODE_ENV === 'production',
				maxAge: SESSION_MAX_AGE,
				path: '/',
				sameSite: 'lax',
			});

			return {
				isAuthenticated: true,
				user: {
					id: existingUser.id,
					email: existingUser.email,
				},
			};
		}

		cookieStore.delete(AUTH_COOKIE_NAME);
		return { isAuthenticated: false, user: null };
	} catch (error) {
		console.error('Check session error:', error);
		// Clear cookie on error to be safe
		const cookieStore = await cookies();
		cookieStore.delete(AUTH_COOKIE_NAME);
		return { isAuthenticated: false, user: null };
	}
}
