'use client';

import type { ReactNode } from 'react';
import { useEffect, useRef } from 'react';
import { usePathname } from 'next/navigation';
import { gsap } from 'gsap';

interface PageTransitionProps {
	children: ReactNode;
}

export function PageTransition({ children }: PageTransitionProps) {
	const pathname = usePathname();
	const contentRef = useRef<HTMLDivElement>(null);
	const curtainTopLeftRef = useRef<HTMLDivElement>(null);
	const curtainBottomRightRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const tl = gsap.timeline();

		// Ensure refs are current before using them
		if (
			curtainTopLeftRef.current &&
			curtainBottomRightRef.current &&
			contentRef.current
		) {
			// Ensure curtains are reset to cover the screen initially and are visible
			gsap.set(curtainTopLeftRef.current, {
				x: '0%',
				y: '0%',
				opacity: 1,
			});
			gsap.set(curtainBottomRightRef.current, {
				x: '0%',
				y: '0%',
				opacity: 1,
			});
			// Content is initially styled with opacity: 0, GSAP ensures it starts at 0 before animating.
			gsap.set(contentRef.current, { opacity: 0 });

			tl.to(curtainTopLeftRef.current, {
				x: '-100%', // Move off-screen towards the top-left
				y: '-100%',
				duration: 0.6,
				ease: 'power2.inOut',
			})
				.to(
					curtainBottomRightRef.current,
					{
						x: '100%', // Move off-screen towards the bottom-right
						y: '100%',
						duration: 0.6,
						ease: 'power2.inOut',
					},
					'<',
				) // The "<" ensures this animation starts at the same time as the previous one
				.to(
					contentRef.current,
					{
						opacity: 1, // Animate content to be visible
						duration: 0.4,
						ease: 'power2.out',
					},
					'-=0.3',
				); // Start content fade-in slightly before curtains fully exit
		}

		// Cleanup function for the timeline to prevent memory leaks
		return () => {
			tl.kill();
		};
	}, [pathname]); // Re-run animation when pathname changes

	return (
		// This outer div persists across route changes.
		<div className="relative min-h-screen">
			{/* Curtain for the top-left part of the slanted split */}
			<div
				ref={curtainTopLeftRef}
				className="fixed inset-0 z-[100] bg-background" // Covers viewport, uses theme background
				style={{ clipPath: 'polygon(0 0, 100% 0, 0 100%)' }} // Defines a top-left triangle
				suppressHydrationWarning // Suppress hydration warning for this element
			/>
			{/* Curtain for the bottom-right part of the slanted split */}
			<div
				ref={curtainBottomRightRef}
				className="fixed inset-0 z-[100] bg-background" // Covers viewport, uses theme background
				style={{ clipPath: 'polygon(100% 100%, 100% 0, 0 100%)' }} // Defines a bottom-right triangle
				suppressHydrationWarning // Suppress hydration warning for this element
			/>
			{/*
        Content container: key={pathname} is crucial.
        It tells React to treat the content as a new component instance on route change.
        Initial opacity is 0 to prevent flash and align with GSAP animation start.
      */}
			<div ref={contentRef} key={pathname} style={{ opacity: 0 }}>
				{children}
			</div>
		</div>
	);
}
