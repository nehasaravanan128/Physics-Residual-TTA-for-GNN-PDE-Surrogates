@echo off
REM ==============================================================================
REM Physics-Residual TTA Interactive Dashboard & API Launcher
REM ==============================================================================

echo [INFO] Starting Physics-Residual TTA Suite...
echo.
echo [1/2] Starting Express API Server on http://localhost:3001
start "MeshGraphNet API Server" cmd /k "cd server && npm start"

echo [2/2] Starting Next.js Web Application on http://localhost:3000
start "MeshGraphNet Web Dashboard" cmd /k "cd webapp && npm run dev"

echo.
echo [DONE] Servers launched!
echo - API Backend: http://localhost:3001
echo - Web Dashboard: http://localhost:3000
echo.
timeout /t 3 /nobreak >nul
start http://localhost:3000
