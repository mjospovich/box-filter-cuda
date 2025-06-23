@echo off
echo === Setting up OpenCV and Compiling CUDA Box Filter ===
echo.

REM Check if OpenCV pre-built exists
if not exist "opencv_prebuilt" (
    echo Downloading OpenCV pre-built for Windows...
    echo This may take a few minutes...
    
    REM Download OpenCV
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-windows.exe' -OutFile 'opencv-4.8.0-windows.exe'}"
    
    if exist "opencv-4.8.0-windows.exe" (
        echo Extracting OpenCV...
        opencv-4.8.0-windows.exe -o"." -y
        rename opencv opencv_prebuilt
        del opencv-4.8.0-windows.exe
        echo OpenCV setup complete!
    ) else (
        echo Failed to download OpenCV. Please check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo OpenCV pre-built found.
)

echo.

REM Create build directory if it doesn't exist
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

echo Setting up Visual Studio environment and compiling...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc code/test.cu -o build/test.exe -I "opencv_prebuilt/build/include" -L "opencv_prebuilt/build/x64/vc16/lib" -lopencv_world480 --std=c++14 -w --diag-suppress=611

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ SUCCESS! Your test.exe is ready in the build directory.
    echo.
    echo Copying OpenCV DLL to build directory...
    copy "opencv_prebuilt\build\x64\vc16\bin\opencv_world480.dll" build\ >nul 2>&1
    echo.
    echo To run: cd build && test.exe
    echo.
) else (
    echo.
    echo ❌ Compilation failed. 
    echo.
)

pause 