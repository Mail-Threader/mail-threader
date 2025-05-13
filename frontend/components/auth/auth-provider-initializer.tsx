"use client";

import { useEffect } from "react";
import { useAuthStore } from "@/store/auth-store";

export function AuthProviderInitializer({
	children,
}: {
	children: React.ReactNode;
}) {
	useEffect(() => {
		// Call checkSession only once on initial mount
		useAuthStore.getState().checkSession();
	}, []);

	return <>{children}</>;
}
