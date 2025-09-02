
# Chatbot (Windows) - Quick Start

1. Open PowerShell in this project folder.
2. Run the setup script (creates venv, installs packages, downloads NLTK & spaCy model):
   ```powershell
   .\setup.ps1
   ```
3. After setup, either:
   - Activate venv and run server:
     ```powershell
     .\venv\Scripts\Activate.ps1
     uvicorn main:app --reload --host 127.0.0.1 --port 8000
     ```
   - Or run server directly with bundled script (no manual activation):
     ```powershell
     .\start_server.bat
     ```
4. Open http://127.0.0.1:8000 in your browser.

Troubleshooting:
- If PowerShell blocks script execution, run as Admin and `Set-ExecutionPolicy RemoteSigned` then retry.
- If NLTK errors, ensure step 2 completed and NLTK data downloaded.
