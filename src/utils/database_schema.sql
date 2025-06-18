-- Database schema for HMI project

-- Table for storing processed emails
CREATE TABLE IF NOT EXISTS processed_emails (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) UNIQUE NOT NULL,
    main_id VARCHAR(255),
    filename VARCHAR(255),
    type VARCHAR(50),
    date TIMESTAMP,
    sender VARCHAR(255),
    x_from VARCHAR(255),
    x_to VARCHAR(255),
    original_sender VARCHAR(255),
    original_date TIMESTAMP,
    recipient VARCHAR(255),
    subject TEXT,
    cc VARCHAR(255),
    x_cc VARCHAR(255),
    body TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing topic analysis results
CREATE TABLE IF NOT EXISTS topic_analysis (
    id SERIAL PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL,
    keywords JSONB,
    topic_distribution JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing cluster analysis results
CREATE TABLE IF NOT EXISTS cluster_analysis (
    id SERIAL PRIMARY KEY,
    cluster_name VARCHAR(255) NOT NULL,
    cluster_size INTEGER,
    common_words JSONB,
    cluster_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing entity analysis results
CREATE TABLE IF NOT EXISTS entity_analysis (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_name VARCHAR(255) NOT NULL,
    frequency INTEGER,
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, entity_name)
);

-- Table for storing sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES processed_emails(id),
    sentiment_type VARCHAR(50),
    sentiment_score FLOAT,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing summarization results
CREATE TABLE IF NOT EXISTS summarization_results (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES processed_emails(id),
    summary_style VARCHAR(50),
    summary_text TEXT,
    word_count INTEGER,
    sentence_count INTEGER,
    entity_count INTEGER,
    action_item_count INTEGER,
    key_information JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing visualization data
CREATE TABLE IF NOT EXISTS visualization_data (
    id SERIAL PRIMARY KEY,
    visualization_type VARCHAR(50) NOT NULL,
    file_url TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing analysis metadata
CREATE TABLE IF NOT EXISTS analysis_metadata (
    id SERIAL PRIMARY KEY,
    analysis_type VARCHAR(50) NOT NULL,
    parameters JSONB,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_processed_emails_message_id ON processed_emails(message_id);
CREATE INDEX IF NOT EXISTS idx_processed_emails_date ON processed_emails(date);
CREATE INDEX IF NOT EXISTS idx_processed_emails_sender ON processed_emails(sender);
CREATE INDEX IF NOT EXISTS idx_processed_emails_recipient ON processed_emails(recipient);
CREATE INDEX IF NOT EXISTS idx_entity_analysis_type_name ON entity_analysis(entity_type, entity_name);
CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_email_id ON sentiment_analysis(email_id);
CREATE INDEX IF NOT EXISTS idx_summarization_results_email_id ON summarization_results(email_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for updating updated_at column
CREATE TRIGGER update_processed_emails_updated_at
    BEFORE UPDATE ON processed_emails
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
