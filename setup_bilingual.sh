#!/bin/bash

# Bilingual Setup Script
# This script helps you set up bilingual support (EN/ES) for your Flask website

echo "🌐 Bilingual Support Setup Script"
echo "=================================="
echo ""

# Check if Flask-Babel is installed
echo "📦 Checking Flask-Babel installation..."
if python -c "import flask_babel" 2>/dev/null; then
    echo "✅ Flask-Babel is already installed"
else
    echo "❌ Flask-Babel is not installed"
    echo "Please run: pip install Flask-Babel"
    exit 1
fi

echo ""
echo "Step 1: Extracting translatable strings..."
pybabel extract -F babel.cfg -o messages.pot .
if [ $? -eq 0 ]; then
    echo "✅ Strings extracted successfully"
else
    echo "❌ Failed to extract strings"
    exit 1
fi

echo ""
echo "Step 2: Initializing Spanish translation..."
if [ -d "app/translations/es" ]; then
    echo "⚠️  Spanish translation already exists, updating instead..."
    pybabel update -i messages.pot -d app/translations -l es
else
    pybabel init -i messages.pot -d app/translations -l es
fi
if [ $? -eq 0 ]; then
    echo "✅ Spanish translation initialized"
else
    echo "❌ Failed to initialize Spanish translation"
    exit 1
fi

echo ""
echo "Step 3: Initializing English translation..."
if [ -d "app/translations/en" ]; then
    echo "⚠️  English translation already exists, updating instead..."
    pybabel update -i messages.pot -d app/translations -l en
else
    pybabel init -i messages.pot -d app/translations -l en
fi
if [ $? -eq 0 ]; then
    echo "✅ English translation initialized"
else
    echo "❌ Failed to initialize English translation"
    exit 1
fi

echo ""
echo "📝 Next steps:"
echo "1. Edit app/translations/es/LC_MESSAGES/messages.po"
echo "2. Add Spanish translations for all msgid entries"
echo "3. Run: pybabel compile -d app/translations"
echo "4. Start your Flask app: python run.py"
echo ""
echo "See BILINGUAL_SETUP.md for detailed instructions and example translations"
