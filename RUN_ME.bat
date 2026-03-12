@echo off
title Resume Ranking System - Running...
color 0A
cls

echo.
echo ====================================================================
echo                    RESUME RANKING SYSTEM
echo                 Starting Analysis - Please Wait...
echo ====================================================================
echo.

cd /d "%~dp0"
python main.py

echo.
echo ====================================================================
echo                    EXECUTION COMPLETE!
echo ====================================================================
echo.
echo Results saved to: rankings_output.csv
echo.
echo Press any key to view the results...
pause > nul

type rankings_output.csv

echo.
echo.
echo Press any key to close...
pause > nul
