@echo off
chcp 65001 >nul
echo Starting both services...
echo.
start "Python RAG Service" cmd /c "%~dp0start-python.bat"
timeout /t 3 /nobreak >nul
start "Spring Boot Backend" cmd /c "%~dp0start-java.bat"
echo.
echo Both services starting:
echo   Python RAG Service: http://127.0.0.1:8001
echo   Spring Boot + Frontend: http://127.0.0.1:8080
