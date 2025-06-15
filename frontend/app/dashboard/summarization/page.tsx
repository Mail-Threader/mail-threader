import { getSummarizedEmails } from '@/lib/queries';
import SummarizationClient from './SummarizationClient';

export default async function SummarizationPage() {
	const { data, pageCount } = await getSummarizedEmails({
		page: 1,
		perPage: 10,
		sort: [],
		filters: [],
		joinOperator: 'and',
		filterFlag: 'advancedFilters',
		subject: '',
	});

	return <SummarizationClient data={data} pageCount={pageCount} />;
}
