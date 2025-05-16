import { Skeleton } from '@/components/ui/skeleton';

export default function Loading() {
	// You can add any UI inside Loading, including a Skeleton.
	return (
		<div className="flex flex-col items-center justify-center min-h-screen p-4">
			<Skeleton className="h-12 w-12 rounded-full mb-4" />
			<Skeleton className="h-4 w-[250px] mb-2" />
			<Skeleton className="h-4 w-[200px]" />
		</div>
	);
}
