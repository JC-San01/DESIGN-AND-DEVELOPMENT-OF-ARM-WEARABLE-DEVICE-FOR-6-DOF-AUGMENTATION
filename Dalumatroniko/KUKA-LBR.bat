@echo off
title KUKA LBR iiwa Robot Controller
echo Starting Robot Simulation...

:: Navigate to your project directory 
cd /d "D:\DOWNLOADS\Dalumatroniko"

:: Activate virtual environment 
call "D:\DOWNLOADS\Dalumatroniko\venv\Scripts\activate.bat"

:: Run the Python script
python SimulateV8.py

:: Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause > nul
)

:: Deactivate virtual environment (optional)
call deactivate

 