@echo off
echo ============================================
echo   Water Quality QC Tool - Starting...
echo ============================================
echo.
echo Opening in your browser at http://localhost:8501
echo Press Ctrl+C in this window to stop the app.
echo ============================================
echo.
cd /d "%~dp0"
streamlit run water_quality_qc_app.py
pause
