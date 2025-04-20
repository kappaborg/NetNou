@echo off
REM NetNou - AI Student Attendance & Engagement Analysis
REM Helper script to run common commands on Windows

REM Change to the directory where the script is located
cd /d "%~dp0"

REM Display header
echo NetNou Helper Script
echo.

REM Function to print usage information
:print_usage
echo Available commands:
echo   analyze          - Run real-time face and engagement analysis
echo   analyze:fast     - Run analysis optimized for speed
echo   analyze:accurate - Run analysis optimized for accuracy
echo   train            - Train the engagement neural network
echo   webapp           - Run the web application
echo   setup            - Install dependencies
echo   help             - Show this help message
echo.
echo Examples:
echo   run_netnou.bat analyze
echo   run_netnou.bat analyze:fast
echo   run_netnou.bat train
goto :eof

REM Check if argument is provided
if "%1"=="" (
    call :print_usage
    exit /b 1
)

REM Process the command
if "%1"=="analyze" (
    echo Starting face and engagement analysis...
    python NetNou\demographic_analysis\live_demographics.py
    goto :end
)

if "%1"=="analyze:fast" (
    echo Starting optimized analysis for speed...
    python NetNou\demographic_analysis\live_demographics.py --detector ssd --analyze_every 3
    goto :end
)

if "%1"=="analyze:accurate" (
    echo Starting optimized analysis for accuracy...
    python NetNou\demographic_analysis\live_demographics.py --detector retinaface --analyze_every 1
    goto :end
)

if "%1"=="train" (
    echo Training engagement neural network...
    python NetNou\scratch_nn\train_engagement_nn.py
    goto :end
)

if "%1"=="webapp" (
    echo Starting web application...
    if exist NetNou-WebApp (
        cd NetNou-WebApp
        python run.py
    ) else (
        echo WebApp directory not found. Have you cloned the repository?
        exit /b 1
    )
    goto :end
)

if "%1"=="setup" (
    echo Installing dependencies...
    echo Creating virtual environment...
    python -m venv venv
    
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo Installing required packages...
    pip install -r requirements.txt
    echo Setup complete! You can now run other commands.
    goto :end
)

if "%1"=="help" (
    call :print_usage
    goto :end
)

echo Unknown command: %1
call :print_usage
exit /b 1

:end
exit /b 0 