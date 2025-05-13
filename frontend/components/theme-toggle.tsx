"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { usePreferencesStore } from "@/store/preferences-store";

export function ThemeToggle() {
	const currentTheme = usePreferencesStore((state) => state.theme);
	const setThemeInStore = usePreferencesStore((state) => state.setTheme);
	const [mounted, setMounted] = React.useState(false);

	React.useEffect(() => {
		setMounted(true);
	}, []);

	React.useEffect(() => {
		if (!mounted) return; // Wait for store to be hydrated

		const root = window.document.documentElement;
		root.classList.remove("theme-light", "dark");

		if (currentTheme === "system") {
			const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
				.matches
				? "dark"
				: "theme-light";
			root.classList.add(systemTheme);
		} else {
			root.classList.add(currentTheme);
		}
	}, [currentTheme, mounted]);

	React.useEffect(() => {
		if (!mounted || currentTheme !== "system") {
			return;
		}

		const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
		const handleChange = () => {
			// This ensures the DOM updates if the system theme changes while 'system' is selected in the app
			const root = window.document.documentElement;
			root.classList.remove("theme-light", "dark");
			const systemTheme = mediaQuery.matches ? "dark" : "theme-light";
			root.classList.add(systemTheme);
		};

		mediaQuery.addEventListener("change", handleChange);
		return () => mediaQuery.removeEventListener("change", handleChange);
	}, [currentTheme, mounted]);

	if (!mounted) {
		// To prevent hydration mismatch, render a placeholder or null until mounted
		// and theme is potentially hydrated from localStorage by Zustand.
		// Returning the button structure helps avoid layout shifts.
		return (
			<Button variant="ghost" size="icon" disabled>
				<Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
				<Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
				<span className="sr-only">Toggle theme (loading)</span>
			</Button>
		);
	}

	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild>
				<Button variant="ghost" size="icon">
					<Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
					<Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
					<span className="sr-only">Toggle theme</span>
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent align="end">
				<DropdownMenuItem onClick={() => setThemeInStore("theme-light")}>
					Light
				</DropdownMenuItem>
				<DropdownMenuItem onClick={() => setThemeInStore("dark")}>
					Dark
				</DropdownMenuItem>
				<DropdownMenuItem onClick={() => setThemeInStore("system")}>
					System
				</DropdownMenuItem>
			</DropdownMenuContent>
		</DropdownMenu>
	);
}
