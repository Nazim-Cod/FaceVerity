@echo off
cd /d %~dp0
call venv\Scripts\activate
set PYTHONPATH=%CD%
echo ================================
echo   FaceVerify TF - Menu
echo ================================
echo 1. Installer les dependances
echo 2. Lancer le fine-tuning
echo 3. Lancer l'evaluation
echo 4. Lancer la GUI
echo ================================
set /p choix="Votre choix (1/2/3/4) : "
if "%choix%"=="1" pip install -r requirements.txt
if "%choix%"=="2" python train.py --model both --epochs 5
if "%choix%"=="3" python evaluate.py
if "%choix%"=="4" streamlit run gui/app.py
pause
