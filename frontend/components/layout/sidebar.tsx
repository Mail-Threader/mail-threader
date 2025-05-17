'use client';

import { ReactNode, useEffect } from 'react';

const Sidebar = ({
	children,
}: Readonly<{
	children: ReactNode;
}>) => {
	useEffect(() => {}, []);

	return (
		<div id="sidebarContainer" className="w-full flex bg-blue-50">
			<div
				id="sidebar"
				className="min-w-[5vw] max-w-[20vw] resize-x overflow-auto bg-yellow-100"
			></div>
			{children}
		</div>
	);
};

export default Sidebar;
