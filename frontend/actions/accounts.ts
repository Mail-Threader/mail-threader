'use server';

import { cookies } from 'next/headers';
import { db } from '@/db';
import { usersTable, type User } from '@/db/schema';
import { eq } from 'drizzle-orm';
import { AUTH_COOKIE_NAME } from '@/lib/utils';
import { checkSessionAction } from './auth';
import { compare, hash } from 'bcrypt';
import {
	ChangePasswordFormInputs,
	changePasswordSchema,
} from '@/schemas/validation-schemas';

export interface UserDetails
	extends Pick<User, 'email' | 'createdAt' | 'updatedAt'> {}

export interface UserDetailsResponse {
	success: boolean;
	user?: UserDetails;
	message?: string;
}

export interface AccountActionResponse {
	success: boolean;
	message?: string;
	errors?: {
		currentPassword?: string[];
		newPassword?: string[];
		confirmNewPassword?: string[];
		_form?: string[];
	};
}

export async function getUserDetailsAction(): Promise<UserDetailsResponse> {
	try {
		const userRes = await checkSessionAction();
		if (!userRes || !userRes.isAuthenticated || !userRes.user) {
			return { success: false, message: 'User not authenticated.' };
		}
		return {
			success: true,
			user: {
				email: userRes.user.email,
				createdAt: userRes.user.createdAt,
				updatedAt: userRes.user.updatedAt,
			},
		};
	} catch (error) {
		console.error('Error fetching user details:', error);
		return { success: false, message: 'Could not fetch account details.' };
	}
}

export async function changePasswordAction(
	data: ChangePasswordFormInputs,
): Promise<AccountActionResponse> {
	const validationResult = changePasswordSchema.safeParse(data);
	if (!validationResult.success) {
		return {
			success: false,
			message: 'Invalid form data.',
			errors: validationResult.error.flatten().fieldErrors,
		};
	}

	const { currentPassword, newPassword } = validationResult.data;

	try {
		const userRes = await checkSessionAction();
		if (!userRes || !userRes.isAuthenticated || !userRes.user) {
			return {
				success: false,
				message: 'User not authenticated.',
				errors: { _form: ['User not authenticated.'] },
			};
		}

		const user = userRes.user;

		const passwordMatch = await compare(currentPassword, user.password);
		if (!passwordMatch) {
			return {
				success: false,
				message: 'Incorrect current password.',
				errors: { currentPassword: ['Incorrect current password.'] },
			};
		}

		const newHashedPassword = await hash(newPassword, 10);

		await db
			.update(usersTable)
			.set({ password: newHashedPassword, updatedAt: new Date() })
			.where(eq(usersTable.id, user.id));

		return { success: true, message: 'Password updated successfully.' };
	} catch (error) {
		console.error('Change password error:', error);
		return {
			success: false,
			message: 'An unexpected error occurred.',
			errors: { _form: ['An unexpected error occurred.'] },
		};
	}
}

export async function deleteAccountAction(): Promise<AccountActionResponse> {
	try {
		const userRes = await checkSessionAction();
		if (!userRes || !userRes.isAuthenticated || !userRes.user) {
			return { success: false, message: 'User not authenticated.' };
		}

		await db.delete(usersTable).where(eq(usersTable.id, userRes.user.id));

		// Clear the authentication cookie
		(await cookies()).delete(AUTH_COOKIE_NAME);

		return { success: true, message: 'Account deleted successfully.' };
	} catch (error) {
		console.error('Delete account error:', error);
		return {
			success: false,
			message: 'An unexpected error occurred while deleting the account.',
		};
	}
}
