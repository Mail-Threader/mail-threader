'use client';

import { ReactNode } from 'react';

const Sidebar = ({
	children,
}: Readonly<{
	children: ReactNode;
}>) => {
	return <div className="">{children}</div>;
};

export default Sidebar;
