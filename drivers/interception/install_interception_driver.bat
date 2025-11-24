@echo off
REM Interception Driver Installer
REM Run this file as Administrator (Right-click -> Run as administrator)

echo ============================================================
echo Interception Driver Installation
echo ============================================================
echo.

cd /d "%~dp0"
set INSTALLER_PATH=interception_driver\Interception\command line installer\install-interception.exe

if not exist "%INSTALLER_PATH%" (
    echo ERROR: Installer not found at: %INSTALLER_PATH%
    echo Please make sure the driver was downloaded and extracted.
    pause
    exit /b 1
)

echo Installing Interception driver...
echo This requires Administrator privileges.
echo.

"%INSTALLER_PATH%" /install

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo Installation successful!
    echo ============================================================
    echo.
    echo IMPORTANT: You must restart your computer for the driver to work.
    echo.
    echo After restart, verify installation by running:
    echo   python scripts\check_interception.py
    echo.
) else (
    echo.
    echo ============================================================
    echo Installation failed or requires restart
    echo ============================================================
    echo.
    echo Please check the error messages above.
    echo You may need to restart your computer and try again.
    echo.
)

pause


