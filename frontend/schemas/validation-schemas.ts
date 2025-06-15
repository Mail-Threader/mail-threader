import { z } from 'zod';

export const loginSchema = z.object({
	email: z.string().email({ message: 'Invalid email address.' }),
	password: z
		.string()
		.min(7, { message: 'Password must be 7 characters' })
		.max(20, {
			message: 'Password must be 20 characters or less',
		})
		.regex(
			/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{7,}$/,
			'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character.',
		),
});

export type LoginFormInputs = z.infer<typeof loginSchema>;

export const signupSchema = z
	.object({
		email: z.string().email({ message: 'Invalid email address.' }),
		password: z
			.string()
			.min(7, { message: 'Password must be at least 6 characters.' })
			.max(20, {
				message: 'Password must be at most 20 characters.',
			})
			.regex(
				/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{7,}$/,
				'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character.',
			),
		confirmPassword: z
			.string()
			.min(6, { message: 'Password must be at least 6 characters.' }),
	})
	.refine((data) => data.password === data.confirmPassword, {
		message: "Passwords don't match",
		path: ['confirmPassword'], // path to field that will display the error
	});

export type SignupFormInputs = z.infer<typeof signupSchema>;

export const changePasswordSchema = z
	.object({
		currentPassword: z
			.string()
			.min(7, { message: 'Password must be 7 characters' })
			.max(20, {
				message: 'Password must be 20 characters or less',
			})
			.regex(
				/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{7,}$/,
				'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character.',
			),
		newPassword: z
			.string()
			.min(7, { message: 'New Password must be 7 characters' })
			.max(20, {
				message: 'New Password must be 20 characters or less',
			})
			.regex(
				/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{7,}$/,
				'New Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character.',
			),
		confirmNewPassword: z
			.string()
			.min(7, { message: 'Confirm Password must be 7 characters' })
			.max(20, {
				message: 'Confirm Password must be 20 characters or less',
			})
			.regex(
				/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{7,}$/,
				'Confirm Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character.',
			),
	})
	.refine((data) => data.newPassword === data.confirmNewPassword, {
		message: "New passwords don't match.",
		path: ['confirmNewPassword'],
	});

export type ChangePasswordFormInputs = z.infer<typeof changePasswordSchema>;
