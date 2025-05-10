-- prepare_similarity_search_service_db.sql

-- Create the database for the similarity search service
CREATE DATABASE images;


-- Skrypty w Dockerze są uruchamiane tylko w bazie "postgres"
-- osobny skrypt
-- Connect to the database
\connect images

-- Enable vectorscale extension
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;