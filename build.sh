#!/usr/bin/env bash
# Exit on error
set -o errexit
set -o nounset  # To exit on unset variables

# Upgrade pip
python -m pip install --upgrade pip

# Install system dependencies for dlib compilation (skip sudo)
# Ensure that any necessary dependencies are already available in the platform environment

# Install dlib - Using precompiled wheels to avoid memory issues (alternative option)
pip install dlib --no-cache-dir

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Create necessary directories
mkdir -p static staticfiles media

# Make fresh migrations
python manage.py makemigrations --noinput

# Apply migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput --clear

# Create superuser (ensure no input errors with default or env variable overrides)
DJANGO_SUPERUSER_EMAIL="${DJANGO_SUPERUSER_EMAIL:-admin@example.com}"
DJANGO_SUPERUSER_USERNAME="${DJANGO_SUPERUSER_USERNAME:-admin}"
DJANGO_SUPERUSER_PASSWORD="${DJANGO_SUPERUSER_PASSWORD:-adminpassword}"

# Avoid creating superuser if the username already exists
python manage.py createsuperuser --noinput \
    --username "$DJANGO_SUPERUSER_USERNAME" \
    --email "$DJANGO_SUPERUSER_EMAIL" || echo "Superuser creation skipped due to existing username."

# Display completion message
echo "Build completed successfully!"
