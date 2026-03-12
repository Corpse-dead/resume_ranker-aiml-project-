@echo off
title Resume Ranking System - Auto Run
color 0A
cls

echo.
echo ====================================================================
echo                    RESUME RANKING SYSTEM
echo                 AI-Powered Resume Analysis
echo ====================================================================
echo.

cd /d "%~dp0"
python main.py

echo.
echo ====================================================================
echo                       FINAL RESULTS
echo ====================================================================
echo.

type rankings_output.csv

echo.
echo ====================================================================
echo           Analysis Complete! Results saved to CSV file.
echo ====================================================================
echo.

timeout /t 10 /nobreak > nul
