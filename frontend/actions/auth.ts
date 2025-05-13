'use server';

import {
	loginSchema,
	signupSchema,
	type LoginFormInputs,
	type SignupFormInputs,
} from '@/schemas/auth-schemas';
import { db } from '@/db';
import { usersTable } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { compare, genSalt, hash } from 'bcrypt';
import { cookies } from 'next/headers';

const AUTH_COOKIE_NAME = 'auth_session_token';
// In a real app, use a proper secret for JWT or session signing
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

export interface AuthState {
	success: boolean;
	message?: string;
	user?: { email: string };
	errors?: {
		email?: string[];
		password?: string[];
		confirmPassword?: string[]; // For signup
		_form?: string[];
	};
}

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
		});

		if (!existingUser) {
			return { success: false, message: 'Invalid email or password.' };
		}

		const passwordMatch = await compare(
			password,
			existingUser.hashedPassword,
		);

		if (!passwordMatch) {
			return { success: false, message: 'Invalid email or password.' };
		}

		const sessionToken = existingUser.email;

		const cookieStore = await cookies();

		cookieStore.set(AUTH_COOKIE_NAME, sessionToken, {
			httpOnly: true,
			secure: process.env.NODE_ENV === 'production',
			maxAge: SESSION_MAX_AGE,
			path: '/',
			sameSite: 'lax',
		});

		return {
			success: true,
			message: 'Login successful!',
			user: { email: existingUser.email },
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

	const { email, password } = validationResult.data;

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
			hashedPassword,
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
	user: { email: string } | null;
}

export async function checkSessionAction(): Promise<CheckSessionResult> {
	try {
		const cookieStore = await cookies();
		const sessionToken = cookieStore.get(AUTH_COOKIE_NAME)?.value;

		if (!sessionToken) {
			return { isAuthenticated: false, user: null };
		}

		// In a real app, you would validate the token (e.g., JWT verification, session DB lookup)
		// For this example, if token exists, assume it's valid and it's the email.
		const existingUser = await db.query.usersTable.findFirst({
			where: eq(usersTable.email, sessionToken),
		});

		if (existingUser) {
			return {
				isAuthenticated: true,
				user: { email: existingUser.email },
			};
		}

		// If token is present but user not found (e.g. token tampered or user deleted), clear cookie
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
