'use client';

import { useEffect, useState, useTransition } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
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
import {
	Form,
	FormControl,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from '@/components/ui/form';
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
	AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/store/auth-store';
import {
	UserCircle2Icon,
	KeyRoundIcon,
	ShieldAlertIcon,
	Loader2Icon,
} from 'lucide-react';
import { format } from 'date-fns';
import { Skeleton } from '@/components/ui/skeleton';
import {
	AccountActionResponse,
	changePasswordAction,
	deleteAccountAction,
	getUserDetailsAction,
	UserDetailsResponse,
} from '@/actions/accounts';
import {
	ChangePasswordFormInputs,
	changePasswordSchema,
} from '@/schemas/validation-schemas';

interface UserDisplayDetails {
	email: string;
	createdAt?: string;
	updatedAt?: string;
}

export default function AccountPage() {
	const { toast } = useToast();
	const router = useRouter();
	const { user, logout } = useAuthStore();

	const [userDetails, setUserDetails] = useState<UserDisplayDetails | null>(
		null,
	);
	const [isLoadingDetails, setIsLoadingDetails] = useState(true);
	const [isPasswordPending, startPasswordTransition] = useTransition();
	const [isDeletePending, startDeleteTransition] = useTransition();
	const [passwordFormError, setPasswordFormError] = useState<string | null>(
		null,
	);

	useEffect(() => {
		const fetchDetails = async () => {
			if (user?.email) {
				setIsLoadingDetails(true);
				const result: UserDetailsResponse =
					await getUserDetailsAction();
				if (result.success && result.user) {
					setUserDetails({
						email: result.user.email,
						createdAt: result.user.createdAt
							? format(new Date(result.user.createdAt), 'PPP')
							: 'N/A',
						updatedAt: result.user.updatedAt
							? format(new Date(result.user.updatedAt), 'PPP p')
							: 'N/A',
					});
				} else {
					toast({
						title: 'Error',
						description:
							result.message ||
							'Could not fetch account details.',
						variant: 'destructive',
					});
					// Fallback to auth store email if DB fetch fails for some reason
					setUserDetails({ email: user.email });
				}
				setIsLoadingDetails(false);
			} else {
				// Should be caught by AuthGuard, but as a fallback:
				router.push('/login');
			}
		};
		fetchDetails();
	}, [user, toast, router]);

	const passwordForm = useForm<ChangePasswordFormInputs>({
		resolver: zodResolver(changePasswordSchema),
		defaultValues: {
			currentPassword: '',
			newPassword: '',
			confirmNewPassword: '',
		},
	});

	const onSubmitPasswordChange = (data: ChangePasswordFormInputs) => {
		setPasswordFormError(null);
		startPasswordTransition(async () => {
			const result: AccountActionResponse = await changePasswordAction(
				data,
			);
			if (result.success) {
				toast({
					title: 'Password Changed',
					description: 'Your password has been updated successfully.',
				});
				passwordForm.reset();
			} else {
				setPasswordFormError(
					result.message || 'Failed to change password.',
				);
				toast({
					title: 'Error',
					description: result.message || 'Failed to change password.',
					variant: 'destructive',
				});
			}
		});
	};

	const handleDeleteAccount = () => {
		startDeleteTransition(async () => {
			const result: AccountActionResponse = await deleteAccountAction();
			if (result.success) {
				logout();
				toast({
					title: 'Account Deleted',
					description: 'Your account has been permanently deleted.',
				});
				router.push('/login');
			} else {
				toast({
					title: 'Error',
					description: result.message || 'Failed to delete account.',
					variant: 'destructive',
				});
			}
		});
	};

	return (
		<div className="space-y-8">
			<Card className="shadow-lg">
				<CardHeader>
					<div className="flex items-center gap-3 mb-2">
						<UserCircle2Icon className="h-8 w-8 text-primary" />
						<CardTitle className="text-2xl">
							Account Information
						</CardTitle>
					</div>
					<CardDescription>
						View and manage your account details.
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					{isLoadingDetails ? (
						<>
							<Skeleton className="h-6 w-3/4" />
							<Skeleton className="h-5 w-1/2" />
							<Skeleton className="h-5 w-1/2" />
						</>
					) : (
						<>
							<div>
								<FormLabel className="text-sm text-muted-foreground">
									Email Address
								</FormLabel>
								<p className="text-lg font-medium">
									{userDetails?.email || 'Loading...'}
								</p>
							</div>
							<div>
								<FormLabel className="text-sm text-muted-foreground">
									Joined On
								</FormLabel>
								<p>{userDetails?.createdAt || 'Loading...'}</p>
							</div>
							<div>
								<FormLabel className="text-sm text-muted-foreground">
									Last Profile Update
								</FormLabel>
								<p>{userDetails?.updatedAt || 'Loading...'}</p>
							</div>
						</>
					)}
				</CardContent>
			</Card>

			<Card className="shadow-lg">
				<CardHeader>
					<div className="flex items-center gap-3 mb-2">
						<KeyRoundIcon className="h-8 w-8 text-primary" />
						<CardTitle className="text-2xl">
							Change Password
						</CardTitle>
					</div>
					<CardDescription>
						Update your account password. Make sure it&apos;s strong
						and unique.
					</CardDescription>
				</CardHeader>
				<Form {...passwordForm}>
					<form
						onSubmit={passwordForm.handleSubmit(
							onSubmitPasswordChange,
						)}
					>
						<CardContent className="space-y-6">
							{passwordFormError && (
								<div className="rounded-md border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
									{passwordFormError}
								</div>
							)}
							<FormField
								control={passwordForm.control}
								name="currentPassword"
								render={({ field }) => (
									<FormItem>
										<FormLabel>Current Password</FormLabel>
										<FormControl>
											<Input
												type="password"
												placeholder="••••••••"
												{...field}
												disabled={isPasswordPending}
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>
							<FormField
								control={passwordForm.control}
								name="newPassword"
								render={({ field }) => (
									<FormItem>
										<FormLabel>New Password</FormLabel>
										<FormControl>
											<Input
												type="password"
												placeholder="••••••••"
												{...field}
												disabled={isPasswordPending}
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>
							<FormField
								control={passwordForm.control}
								name="confirmNewPassword"
								render={({ field }) => (
									<FormItem>
										<FormLabel>
											Confirm New Password
										</FormLabel>
										<FormControl>
											<Input
												type="password"
												placeholder="••••••••"
												{...field}
												disabled={isPasswordPending}
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>
						</CardContent>
						<CardFooter>
							<Button type="submit" disabled={isPasswordPending}>
								{isPasswordPending && (
									<Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
								)}
								Change Password
							</Button>
						</CardFooter>
					</form>
				</Form>
			</Card>

			<Card className="border-destructive shadow-lg">
				<CardHeader>
					<div className="flex items-center gap-3 mb-2">
						<ShieldAlertIcon className="h-8 w-8 text-destructive" />
						<CardTitle className="text-2xl text-destructive">
							Danger Zone
						</CardTitle>
					</div>
					<CardDescription className="text-destructive/90">
						Proceed with caution. These actions are irreversible.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<div className="space-y-4">
						<div>
							<h4 className="font-semibold text-lg">
								Delete Account
							</h4>
							<p className="text-sm text-muted-foreground">
								Permanently delete your account and all
								associated data. This action cannot be undone.
							</p>
						</div>
						<AlertDialog>
							<AlertDialogTrigger asChild>
								<Button
									variant="destructive"
									disabled={isDeletePending}
								>
									{isDeletePending && (
										<Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
									)}
									Delete My Account
								</Button>
							</AlertDialogTrigger>
							<AlertDialogContent>
								<AlertDialogHeader>
									<AlertDialogTitle>
										Are you absolutely sure?
									</AlertDialogTitle>
									<AlertDialogDescription>
										This action cannot be undone. This will
										permanently delete your account and
										remove your data from our servers.
									</AlertDialogDescription>
								</AlertDialogHeader>
								<AlertDialogFooter>
									<AlertDialogCancel
										disabled={isDeletePending}
									>
										Cancel
									</AlertDialogCancel>
									<AlertDialogAction
										onClick={handleDeleteAccount}
										disabled={isDeletePending}
										className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
									>
										{isDeletePending && (
											<Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
										)}
										Yes, delete my account
									</AlertDialogAction>
								</AlertDialogFooter>
							</AlertDialogContent>
						</AlertDialog>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
