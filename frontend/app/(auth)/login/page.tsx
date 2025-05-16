'use client';

import { Button } from '@/components/ui/button';
import {
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useForm } from 'react-hook-form';
import Link from 'next/link';
import {
	Form,
	FormField,
	FormItem,
	FormLabel,
	FormControl,
	FormMessage,
} from '@/components/ui/form';
import { MailThreaderLogo } from '@/components/icons/mail-threader-logo';
import { loginAction, type AuthState } from '@/actions/auth';
import { zodResolver } from '@hookform/resolvers/zod';
import {
	loginSchema,
	type LoginFormInputs,
} from '@/schemas/validation-schemas';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';
import { Fragment, useState, useTransition } from 'react';
import { useAuthStore } from '@/store/auth-store';

export default function LoginPage() {
	const { toast } = useToast();
	const router = useRouter();
	const [isPending, startTransition] = useTransition();
	const [formError, setFormError] = useState<string | null>(null);
	const authStoreLogin = useAuthStore((state) => state.login);

	const form = useForm<LoginFormInputs>({
		resolver: zodResolver(loginSchema),
		defaultValues: {
			email: '',
			password: '',
		},
	});

	const onSubmit = (data: LoginFormInputs) => {
		setFormError(null);
		startTransition(async () => {
			const result: AuthState = await loginAction(data);
			if (result.success && result.user) {
				authStoreLogin(result.user); // Update Zustand store for immediate UI feedback
				toast({
					title: 'Login Successful',
					description: 'Welcome back!',
				});
				router.push('/dashboard/data-view'); // Cookie is set by server action
			} else {
				if (result.errors?._form) {
					setFormError(result.errors._form.join(', '));
				} else if (result.message) {
					setFormError(result.message);
				} else {
					setFormError(
						'Login failed. Please check your credentials.',
					);
				}

				// Handle field-specific errors if any
				if (result.errors) {
					Object.entries(result.errors).forEach(([key, value]) => {
						if (value && value.length > 0 && key !== '_form') {
							form.setError(key as keyof LoginFormInputs, {
								message: value.join(', '),
							});
						}
					});
				}

				toast({
					title: 'Login Failed',
					description:
						formError ||
						result.message ||
						'Please check your credentials.',
					variant: 'destructive',
				});
			}
		});
	};

	return (
		<Fragment>
			<CardHeader className="space-y-1 text-center">
				<div className="flex justify-center items-center mb-4">
					<MailThreaderLogo className="h-10 w-10 text-primary" />
				</div>
				<CardTitle className="text-2xl">
					Login to Mail-Threader
				</CardTitle>
				<CardDescription>
					Enter your email and password to access your account.
				</CardDescription>
			</CardHeader>
			<Form {...form}>
				<form onSubmit={form.handleSubmit(onSubmit)}>
					<CardContent className="space-y-4">
						{formError && (
							<div className="rounded-md border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
								{formError}
							</div>
						)}
						<FormField
							control={form.control}
							name="email"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Email</FormLabel>
									<FormControl>
										<Input
											placeholder="john@doe.com"
											{...field}
											disabled={isPending}
										/>
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
						<FormField
							control={form.control}
							name="password"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Password</FormLabel>
									<FormControl>
										<Input
											type="password"
											placeholder="*********"
											{...field}
											disabled={isPending}
										/>
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
					</CardContent>
					<CardFooter className="flex flex-col gap-4">
						<Button
							type="submit"
							className="w-full"
							disabled={isPending}
						>
							{isPending ? 'Logging in...' : 'Login'}
						</Button>
						<p className="text-center text-sm text-muted-foreground">
							Don&apos;t have an account?{' '}
							<Link
								href="/signup"
								className="font-semibold text-primary hover:underline"
							>
								Sign up
							</Link>
						</p>
					</CardFooter>
				</form>
			</Form>
		</Fragment>
	);
}
