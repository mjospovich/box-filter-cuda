@echo off
echo.
echo ===============================================================================
echo                           CUDA BOX FILTER COMPILER
echo ===============================================================================
echo.

REM Check if OpenCV pre-built exists
echo [1/3] Checking OpenCV dependencies...
if not exist "opencv_prebuilt" (
    echo       OpenCV not found. Downloading OpenCV 4.8.0...
    echo       This may take a few minutes...
    echo.
    
    REM Download OpenCV
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-windows.exe' -OutFile 'opencv-4.8.0-windows.exe'}"
    
    if exist "opencv-4.8.0-windows.exe" (
        echo       Extracting OpenCV...
        opencv-4.8.0-windows.exe -o"." -y >nul 2>&1
        rename opencv opencv_prebuilt >nul 2>&1
        del opencv-4.8.0-windows.exe >nul 2>&1
        echo       [OK] OpenCV setup complete
    ) else (
        echo       [ERROR] Failed to download OpenCV
        pause
        exit /b 1
    )
) else (
    echo       [OK] OpenCV found
)

echo.
echo [2/3] Preparing build environment...
if not exist "build" (
    echo       Creating build directory...
    mkdir build >nul 2>&1
)
echo       [OK] Build environment ready

echo.
echo [3/3] Compiling CUDA program...
echo       Source: code/box_filter.cu
echo       Target: build/box_filter.exe
echo       Compiler: NVCC + Visual Studio 2022

nvcc code/box_filter.cu -o build/box_filter.exe ^
    -I "opencv_prebuilt/build/include" ^
    -L "opencv_prebuilt/build/x64/vc16/lib" ^
    -lopencv_world480 ^
    --std=c++14 ^
    -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64" ^
    -w --diag-suppress=611 >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo       [OK] Compilation successful
    echo.
    echo       Setting up runtime environment...
    copy "opencv_prebuilt\build\x64\vc16\bin\opencv_world480.dll" build\ >nul 2>&1
    echo       [OK] Runtime setup complete
    echo.
    echo ===============================================================================
    echo                                  SUCCESS
    echo ===============================================================================
    echo.
    echo   Execute: cd build ^&^& box_filter.exe
    echo   Output:  ../results/
    echo   Modes:   Color ^& Grayscale processing
    echo.
) else (
    echo       [ERROR] Compilation failed
    echo.
    echo ===============================================================================
    echo                                  FAILED
    echo ===============================================================================
    echo.
    echo   Check:
    echo   - CUDA Toolkit installed
    echo   - Visual Studio 2022 installed
    echo   - Paths are correct
    echo.
)

pause 