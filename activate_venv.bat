@echo off
REM Activation script for venv
echo Activating virtual environment: venv
call venv\Scripts\activate.bat
echo Virtual environment activated!
echo Python path: %PYTHONPATH%
echo Pip path: where pip
