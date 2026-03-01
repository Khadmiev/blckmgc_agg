#!/bin/sh
set -e

echo "Waiting for PostgreSQL..."
for i in $(seq 1 30); do
    if alembic upgrade head 2>/dev/null; then
        echo "Migrations complete."
        break
    fi
    echo "Attempt $i: waiting for database..."
    sleep 2
done

echo "Starting application..."
exec "$@"
