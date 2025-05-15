'use client';

import { Button } from '@/components/ui/button';
import {
	Card,
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
import { signupAction } from '@/actions/auth';
import { zodResolver } from '@hookform/resolvers/zod';
import { signupSchema, type SignupFormInputs } from '@/schemas/auth-schemas';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';
import { useState, useTransition } from 'react';

export default function SignupPage() {
	const { toast } = useToast();
	const router = useRouter();
	const [isPending, startTransition] = useTransition();
	const [formError, setFormError] = useState<string | null>(null);

	const form = useForm<SignupFormInputs>({
		resolver: zodResolver(signupSchema),
		defaultValues: {
			email: '',
			password: '',
			confirmPassword: '',
		},
	});

	const onSubmit = (data: SignupFormInputs) => {
		setFormError(null);
		startTransition(async () => {
			const result = await signupAction(data);
			if (result.success) {
				toast({
					title: 'Signup Successful',
					description: 'Your account has been created. Please login.',
				});
				router.push('/login');
			} else {
				if (result.errors) {
					Object.entries(result.errors).forEach(([key, value]) => {
						if (value && value.length > 0) {
							form.setError(key as keyof SignupFormInputs, {
								message: value.join(', '),
							});
						}
					});
				}
				if (result.message) {
					setFormError(result.message);
					toast({
						title: 'Signup Failed',
						description: result.message,
						variant: 'destructive',
					});
				}
			}
		});
	};

	return (
		<div className="flex min-h-screen items-center justify-center bg-background p-4">
			<Card className="w-full max-w-md shadow-xl">
				<CardHeader className="space-y-1 text-center">
					<div className="flex justify-center items-center mb-4">
						<MailThreaderLogo className="h-10 w-10 text-primary" />
					</div>
					<CardTitle className="text-2xl">
						Create an Account
					</CardTitle>
					<CardDescription>
						Enter your details to get started with Mail-Threader.
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
							<FormField
								control={form.control}
								name="confirmPassword"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Confirm Password</FormLabel>
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
								{isPending ? 'Signing up...' : 'Sign Up'}
							</Button>
							<p className="text-center text-sm text-muted-foreground">
								Already have an account?{' '}
								<Link
									href="/login"
									className="font-semibold text-primary hover:underline"
								>
									Login
								</Link>
							</p>
						</CardFooter>
					</form>
				</Form>
			</Card>
		</div>
	);
}
