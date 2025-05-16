import { create } from 'zustand';
import { checkSessionAction, type CheckSessionResult } from '@/actions/auth';
import { devtools } from 'zustand/middleware';

interface AuthUser {
	email: string;
	id: string;
}

interface AuthState {
	user: AuthUser | null;
	isAuthenticated: boolean;
	isLoading: boolean; // To track if session check is in progress
	login: (user: AuthUser) => void;
	logout: () => void;
	checkSession: () => Promise<void>;
	setLoading: (status: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
	devtools((set, get) => ({
		user: null,
		isAuthenticated: true,
		isLoading: true, // Start in loading state until session is checked
		login: (user) => {
			set({ user, isAuthenticated: true });
		},
		logout: () => {
			set({ user: null, isAuthenticated: false });
		},
		setLoading: (status: boolean) => {
			set({ isLoading: status });
		},
		checkSession: async () => {
			if (!get().isLoading) {
				// Only set loading if not already loading (e.g. manual call)
				set({ isLoading: true });
			}
			try {
				const sessionResult: CheckSessionResult =
					await checkSessionAction();
				if (sessionResult.isAuthenticated && sessionResult.user) {
					set({
						user: sessionResult.user,
						isAuthenticated: true,
						isLoading: false,
					});
				} else {
					set({
						user: null,
						isAuthenticated: false,
						isLoading: false,
					});
				}
			} catch (error) {
				console.error('Failed to check session:', error);
				set({ user: null, isAuthenticated: false, isLoading: false });
			}
		},
	})),
);
