@echo off
cd /d "%~dp0"
echo Starting Lisbon City Hotel dashboard...
echo Keep this window open while using the dashboard.
echo.
".venv\Scripts\python.exe" -m streamlit run "dashboard.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
echo.
echo The dashboard stopped or failed to start.
pause
