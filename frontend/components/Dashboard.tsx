import React from 'react';
import { Visualization } from './Visualization';
import { Search } from './Search';
import { EmailData } from '../lib/api';

interface DashboardProps {
    data: EmailData[];
}

export const Dashboard: React.FC<DashboardProps> = ({ data }) => {
    return (
        <div className="dashboard">
            <h1>Email Analysis Dashboard</h1>
            <div className="dashboard-grid">
                <div className="search-section">
                    <Search />
                </div>
                <div className="visualization-section">
                    <Visualization data={data} />
                </div>
            </div>
        </div>
    );
};
