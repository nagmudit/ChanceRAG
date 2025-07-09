# ChanceRAG Streamlit Application Setup Script
# This script creates a virtual environment, installs requirements, and runs the app

Write-Host "🤖 ChanceRAG Streamlit Application" -ForegroundColor Cyan
Write-Host "=" -repeat 40 -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.7+ and try again." -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "app.py")) {
    Write-Host "❌ app.py not found. Please run this script from the project directory." -ForegroundColor Red
    exit 1
}

# Prompt for OpenAI API key if not set
if (-not $env:OPENAI_API_KEY) {
    Write-Host "⚠️  OpenAI API key not found in environment variables" -ForegroundColor Yellow
    Write-Host "Please enter your OpenAI API key (or press Enter to skip):" -ForegroundColor Yellow
    $apiKey = Read-Host -AsSecureString
    if ($apiKey.Length -gt 0) {
        $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey))
        Write-Host "✅ API key set for this session" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🚀 Starting setup and application..." -ForegroundColor Cyan

# Run the Python setup script
try {
    python run_app.py
} catch {
    Write-Host "❌ Error running the application: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "👋 Thank you for using ChanceRAG!" -ForegroundColor Cyan
