@echo off
cd /d C:\STOCK\agent_lab
C:\Users\蔡昀達\AppData\Local\uv\uv.exe run --with "finlab>=1.5.9" --with requests python dashboard\generate_dashboard.py >> dashboard\update.log 2>&1
echo [%date% %time%] Done >> dashboard\update.log
