@echo off
chcp 65001 >nul
echo ============================================================
echo   FIRE DETECTION CAMERA - PIPELINE TRAIN
echo ============================================================
echo.

:: Buoc 1: Cai dependencies
echo [1/3] Cai dat thu vien...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [LOI] Cai thu vien that bai!
    pause
    exit /b 1
)
echo [OK] Da cai xong thu vien.
echo.

:: Buoc 2: Train model
echo [2/3] Bat dau train model SVM...
echo ============================================================
python -m src.train
if %errorlevel% neq 0 (
    echo [LOI] Train that bai!
    pause
    exit /b 1
)
echo.

:: Buoc 3: Evaluate
echo [3/3] Danh gia model...
echo ============================================================
python -m src.evaluate
if %errorlevel% neq 0 (
    echo [LOI] Evaluate that bai!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   HOAN THANH! Kiem tra:
echo   - models/svm_fire_detector.pkl
echo   - models/scaler.pkl
echo   - reports/figures/confusion_matrix.png
echo   - reports/figures/metrics_bar.png
echo   - reports/figures/risk_analysis.png
echo ============================================================
echo.
echo   De chay nhan dien realtime tu camera:
echo   python -m src.camera_realtime
echo ============================================================
pause
