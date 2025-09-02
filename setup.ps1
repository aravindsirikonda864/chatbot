
# Run this in PowerShell from the project folder (you may need to run as Administrator to change execution policy once)
# Usage: .\setup.ps1
$ErrorActionPreference = "Stop"

if (-Not (Test-Path venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

# Activate venv for the rest of the script
$venvPython = Join-Path -Path (Get-Location) -ChildPath "venv\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Error "Python executable in venv not found. Ensure Python is installed and accessible."
    exit 1
}

Write-Host "Installing required packages... (this may take a few minutes)"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "Downloading NLTK data (punkt, stopwords)..."
& $venvPython -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Optional: download spaCy model if spacy installed
try {
    & $venvPython -c "import spacy; print('spaCy installed')"
    Write-Host "Downloading spaCy model 'en_core_web_sm'..."
    & $venvPython -m spacy download en_core_web_sm
} catch {
    Write-Host "spaCy not installed, skipping spaCy model download."
}

Write-Host "`nSetup complete. To run the server:"
Write-Host "1) Activate the venv: .\venv\Scripts\Activate.ps1"
Write-Host "2) Start the server: uvicorn main:app --reload --host 127.0.0.1 --port 8000"
