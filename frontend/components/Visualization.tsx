import React from 'react';
import { Line, Bar } from 'react-chartjs-2';
import { EmailData } from '../lib/api';

interface VisualizationProps {
    data: EmailData[];
}

export const Visualization: React.FC<VisualizationProps> = ({ data }) => {
    const sentimentData = {
        labels: data.map((_, i) => `Email ${i + 1}`),
        datasets: [
            {
                label: 'Sentiment Score',
                data: data.map(email => email.sentimentScore || 0),
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
            },
        ],
    };

    const wordFreqData = {
        labels: Object.keys(data[0]?.wordFrequencies || {}),
        datasets: [
            {
                label: 'Word Frequency',
                data: Object.values(data[0]?.wordFrequencies || {}),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 1,
            },
        ],
    };

    return (
        <div className="visualization">
            <div className="chart-container">
                <h2>Sentiment Analysis</h2>
                <Line data={sentimentData} />
            </div>
            <div className="chart-container">
                <h2>Word Frequency</h2>
                <Bar data={wordFreqData} />
            </div>
        </div>
    );
};
