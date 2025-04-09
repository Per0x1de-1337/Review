#!/bin/bash

# Check the server status
curl -s -o /dev/null -w "%{http_code}" http://localhost:5000
echo "Server status dchecked."

# Add a new book via POST request
echo "Adding a new book..."
curl -s -X POST http://localhost:5000/books/ \
-H "Content-Type: application/json" \
-d '{"title": "1984", "author": "George Orwell"}'
echo "Book added."

# Get the list of books with pagination
echo "Fetching books list (page 1, limit 10)..."
curl -s -X GET "http://localhost:5000/books/?page=1&limit=10"
echo "Books fetched."

# Update the book with ID 1
echo "Updating book with ID 1..."
curl -s -X PUT http://localhost:5000/books/1 \
-H "Content-Type: application/json" \
-d '{"title": "1984 - Updated", "author": "George Orwell"}'
echo "Book updated."
