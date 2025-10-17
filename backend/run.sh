#!/bin/bash

echo "starting badminton speed analyzer..."

if [ ! -d "venv" ]; then
    echo "no venv found, creating one..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "venv/installed" ]; then
    echo "installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
fi

echo "server starting on port 8000"
python -m app.main

