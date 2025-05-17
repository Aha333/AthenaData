
# run the setup.: This will install PostgreSQL, start the service, and create the logs_db database.
chmod +x setup.sh
./setup.sh

# connect PostgreSQL: This will create the table log_data and import data from all three text files into PostgreSQL.

psql -d logs_db
# Import the data using the import_data.sql script:
\i /path/to/import_data.sql

