@echo off
chcp 65001 >nul
echo ========================================
echo   Starting Spring Boot Backend (8080)
echo ========================================
echo.
cd /d "%~dp0rag-backend"
call mvnw.cmd spring-boot:run 2>nul || mvn spring-boot:run
pause
