-- This script creates a table and imports data into PostgreSQL.

-- Connect to logs_db
\c logs_db;

-- Create the log_data table
CREATE TABLE IF NOT EXISTS log_data_v2 (
    item_id VARCHAR(255),
    user_id VARCHAR(255),
    action VARCHAR(255),
    vtime TIMESTAMP
);

-- COPY log_data_example(item_id, user_id, action, vtime)
-- FROM '/Users/daweideng/Downloads/tianchi_2014002_rec_tmall_log_samples.txt' DELIMITER E'\u0001' CSV;

--Import data from log_parta.txt, note this using special DELIMITER \u0001
COPY log_data_v2(item_id, user_id, action, vtime)
FROM '/Users/daweideng/Downloads/tianchi_2014002_rec_tmall_log_parta.txt' DELIMITER E'\u0001' CSV;

-- Import data from log_partb.txt
--COPY log_data(item_id, user_id, action, vtime)
--FROM '/Users/daweideng/Downloads/tianchi_2014002_rec_tmall_log_partb.txt' DELIMITER E'\u0001' CSV;


-- Verify the import
SELECT COUNT(*) FROM log_data_v2;

-- Sample query to check data
-- SELECT item_id, COUNT(*) FROM log_data_v2 GROUP BY item_id LIMIT 10;