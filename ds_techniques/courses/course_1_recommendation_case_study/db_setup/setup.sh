#!/bin/bash

# This script installs PostgreSQL, starts the service, creates the database, and prepares for data import.

echo "Installing PostgreSQL..."
# Install PostgreSQL using Homebrew
brew install postgresql

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
brew services start postgresql

# Wait for PostgreSQL to start
sleep 5

# Create database
echo "Creating database 'logs_db'..."
psql postgres -c "CREATE DATABASE logs_db;"

echo "Database 'logs_db' created."

# Set up PostgreSQL connection
echo "Configuring PostgreSQL connection..."
echo "You can now connect to the database using the following command:"
echo "psql -d logs_db"

echo "Installation and database creation completed."