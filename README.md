# Intelligent-Research-Assistant

A Flask-based backend application for the Intelligent Research Assistant.

## Prerequisites
- Python 3.x

## Setup Instructions

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Update dependencies** (after installing new packages):
   ```bash
   pip freeze > requirements.txt
   ```

## Running the Application

Start the Flask development server:
```bash
python app.py
```
Or alternatively:
```bash
flask run
```

The application will be accessible at `http://127.0.0.1:5000/`.

## Project Structure
- `app.py`: Main Flask application entry point.
- `requirements.txt`: Python package dependencies.