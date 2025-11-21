# Bilingual Support Setup Guide (EN/ES)

This guide will help you complete the bilingual support setup for your Flask website.

## ✅ What's Already Done

1. ✅ Flask-Babel configuration added to `config.py`
2. ✅ Babel initialized in `app/__init__.py` with locale selector
3. ✅ Language switching route added (`/set_language/<language>`)
4. ✅ Language switcher UI added to `base.html` (top-right corner)
5. ✅ `babel.cfg` configuration file created
6. ✅ Example translations added to `index.html`

## 📋 Steps to Complete

### Step 1: Install Flask-Babel

```bash
pip install Flask-Babel
```

### Step 2: Extract Translatable Strings

Run this command from the project root directory to extract all translatable strings:

```bash
pybabel extract -F babel.cfg -o messages.pot .
```

This creates a `messages.pot` file with all strings marked for translation.

### Step 3: Initialize Spanish Translation

Create the Spanish translation catalog:

```bash
pybabel init -i messages.pot -d app/translations -l es
```

This creates `app/translations/es/LC_MESSAGES/messages.po`

### Step 4: Initialize English Translation (optional but recommended)

```bash
pybabel init -i messages.pot -d app/translations -l en
```

This creates `app/translations/en/LC_MESSAGES/messages.po`

### Step 5: Translate the Strings

Open `app/translations/es/LC_MESSAGES/messages.po` and add Spanish translations:

Example:
```po
#: app/templates/index.html:6
msgid "Technical Portfolio"
msgstr "Portafolio Técnico"

#: app/templates/index.html:13
msgid "Data Science"
msgstr "Ciencia de Datos"

#: app/templates/index.html:14
msgid "Web Development"
msgstr "Desarrollo Web"

#: app/templates/index.html:20
msgid "About Me"
msgstr "Sobre Mí"

#: app/templates/index.html:22
msgid "I'm"
msgstr "Soy"

#: app/templates/index.html:22
msgid "PhD in Astrophysics from"
msgstr "Doctor en Astrofísica de la"

#: app/templates/index.html:23
msgid "specializing in"
msgstr "especializado en"

#: app/templates/index.html:24
msgid "large-scale data analysis"
msgstr "análisis de datos a gran escala"

#: app/templates/index.html:25
msgid "high-performance image processing and simulations to reconstruct astrophysical environments."
msgstr "procesamiento de imágenes de alto rendimiento y simulaciones para reconstruir entornos astrofísicos."

#: app/templates/index.html:28
msgid "I transform data into clear insights through Machine Learning, automation pipelines, and dashboard storytelling — integrating analytics into real digital products."
msgstr "Transformo datos en insights claros a través de Machine Learning, pipelines de automatización y storytelling con dashboards — integrando analítica en productos digitales reales."

#: app/templates/index.html:42
msgid "Scientific & Technical Foundation"
msgstr "Fundamentos Científicos y Técnicos"

#: app/templates/index.html:44
msgid "Expert in handling massive datasets, image reduction, and reproducible simulations — especially for star cluster modelling."
msgstr "Experto en manejo de conjuntos de datos masivos, reducción de imágenes y simulaciones reproducibles — especialmente para modelado de cúmulos estelares."

#: app/templates/index.html:49
msgid "Data Science & Analytics Expertise"
msgstr "Experiencia en Ciencia de Datos y Analítica"

#: app/templates/index.html:51
msgid "Full pipeline experience: ingestion → preprocessing → ML modelling → insights delivery."
msgstr "Experiencia completa en pipelines: ingestión → preprocesamiento → modelado ML → entrega de insights."

#: app/templates/index.html:56
msgid "Tech Stack"
msgstr "Stack Tecnológico"

#: app/templates/index.html:59
msgid "Machine Learning"
msgstr "Aprendizaje Automático"

#: app/templates/index.html:64
msgid "Data Engineering + Programming"
msgstr "Ingeniería de Datos + Programación"

#: app/templates/index.html:69
msgid "Visualization & BI"
msgstr "Visualización e Inteligencia de Negocios"

#: app/templates/index.html:74
msgid "Web Development"
msgstr "Desarrollo Web"

#: app/templates/index.html:79
msgid "Software Engineering"
msgstr "Ingeniería de Software"

#: app/templates/index.html:85
msgid "Purpose & Motivation"
msgstr "Propósito y Motivación"

#: app/templates/index.html:87
msgid "Committed to applying science and analytics to drive impactful solutions and innovation in Chile and beyond."
msgstr "Comprometido con aplicar ciencia y analítica para impulsar soluciones impactantes e innovación en Chile y más allá."

#: app/templates/index.html:92
msgid "Let's talk — connect on"
msgstr "Hablemos — conéctate en"

#: app/templates/index.html:94
msgid "or"
msgstr "o"
```

### Step 6: Compile Translations

After adding all translations, compile them:

```bash
pybabel compile -d app/translations
```

This creates `.mo` files that Flask-Babel uses.

### Step 7: Test the Implementation

1. Start your Flask server:
   ```bash
   python run.py
   ```

2. Visit `http://127.0.0.1:5000/`

3. Click the **EN/ES** button in the top-right corner to switch languages

## 🔄 Updating Translations

When you add new translatable strings to your templates:

1. Extract new strings:
   ```bash
   pybabel extract -F babel.cfg -o messages.pot .
   ```

2. Update existing catalogs:
   ```bash
   pybabel update -i messages.pot -d app/translations
   ```

3. Edit the `.po` files to add new translations

4. Compile:
   ```bash
   pybabel compile -d app/translations
   ```

## 📝 How to Mark Strings for Translation

### In Templates (Jinja2):

```html
<!-- Simple text -->
<h1>{{ _('Hello World') }}</h1>

<!-- With variables -->
<p>{{ _('Welcome, %(name)s!', name=user.name) }}</p>

<!-- In attributes -->
<img alt="{{ _('Profile picture') }}" src="...">
```

### In Python Code:

```python
from flask_babel import gettext as _

# Simple translation
message = _('Hello World')

# With variables
message = _('Welcome, %(name)s!', name=user.name)
```

## 🎨 Language Switcher

The language switcher is already added to `base.html` and appears in the top-right corner of all pages. It:
- Shows EN/ES buttons
- Highlights the current language
- Stores preference in session and cookie (persists for 1 year)
- Redirects back to the current page after switching

## 📂 File Structure

```
website_ciencia_datos_dev/
├── app/
│   ├── __init__.py          # Babel initialization
│   ├── routes.py            # Language switching route
│   ├── templates/
│   │   ├── base.html        # Language switcher UI
│   │   └── index.html       # Example with translations
│   └── translations/        # Created after pybabel init
│       ├── en/
│       │   └── LC_MESSAGES/
│       │       ├── messages.po
│       │       └── messages.mo
│       └── es/
│           └── LC_MESSAGES/
│               ├── messages.po
│               └── messages.mo
├── babel.cfg                # Babel configuration
└── config.py                # Babel settings
```

## 🚀 Quick Start Commands

```bash
# 1. Install Flask-Babel
pip install Flask-Babel

# 2. Extract, initialize, translate, and compile
pybabel extract -F babel.cfg -o messages.pot .
pybabel init -i messages.pot -d app/translations -l es
pybabel init -i messages.pot -d app/translations -l en

# 3. Edit app/translations/es/LC_MESSAGES/messages.po with Spanish translations

# 4. Compile
pybabel compile -d app/translations

# 5. Run the app
python run.py
```

## 🌐 Supported Languages

- **English (en)**: Default language
- **Spanish (es)**: Secondary language

## 📌 Notes

- The language preference is stored in both session and cookie
- Cookie persists for 1 year
- Falls back to browser's preferred language if no preference is set
- Website files in `website_files/` folder are NOT translated (as requested)

## 🎯 Next Steps

1. Install Flask-Babel
2. Run the pybabel commands
3. Add Spanish translations to the `.po` file
4. Compile and test!

Good luck! 🚀
