@echo off
setlocal EnableDelayedExpansion

REM dependencies.bat - Build dependencies for QuaternionICP on Windows
REM Builds Eigen3, Catch2, Abseil, Ceres, Sophus, args, miniz, tinyexr, SIMDMath, GridSearch

REM ================================================
REM Check if Visual Studio environment is already set
REM ================================================
IF NOT DEFINED VCINSTALLDIR (
    echo Calling vcvarsall.bat for x64...
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 2>nul
    if errorlevel 1 (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 2>nul
    )
    if errorlevel 1 (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 2>nul
    )
    if errorlevel 1 (
        echo ERROR: Could not find Visual Studio 2022 installation.
        echo Please install Visual Studio 2022 with C++ workload.
        exit /b 1
    )
) ELSE (
    echo Visual Studio environment already configured.
)

REM ================================================
REM Parse arguments
REM ================================================
set TOOLSET=msvc
set CONFIG=
set CLEAN_BUILD=
set CI_MODE=
set CMAKE_EXTRA_ARGS=

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--clean" (
    set CLEAN_BUILD=1
    shift
    goto parse_args
)
if /i "%~1"=="--ci" (
    set CI_MODE=1
    shift
    goto parse_args
)
if /i "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="msvc" (
    set TOOLSET=msvc
    shift
    goto parse_args
)
if /i "%~1"=="clangcl" (
    set TOOLSET=clangcl
    shift
    goto parse_args
)
if /i "%~1"=="msvc-asan" (
    set TOOLSET=msvc-asan
    shift
    goto parse_args
)
REM Assume it's a CMake argument
set CMAKE_EXTRA_ARGS=!CMAKE_EXTRA_ARGS! %~1
shift
goto parse_args

:args_done

REM ================================================
REM In CI mode, disable private repo dependencies
REM ================================================
if defined CI_MODE (
    set CMAKE_EXTRA_ARGS=!CMAKE_EXTRA_ARGS! -DBUILD_SIMDMATH=OFF -DBUILD_GRIDSEARCH=OFF
)

REM ================================================
REM Validate toolset
REM ================================================
if /i not "%TOOLSET%"=="msvc" if /i not "%TOOLSET%"=="clangcl" if /i not "%TOOLSET%"=="msvc-asan" (
    echo ERROR: Unknown toolset '%TOOLSET%'. Use 'msvc', 'clangcl', or 'msvc-asan'.
    exit /b 1
)

REM ================================================
REM Set directories
REM ================================================
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set DEP_DIR=%SCRIPT_DIR%\NativeDeps
set INSTALL_PREFIX=%SCRIPT_DIR%\Windows\%TOOLSET%\install
set BUILD_DIR=%SCRIPT_DIR%\Windows\%TOOLSET%\build

echo ==============================================
echo QuaternionICP Dependencies Builder
echo ==============================================
echo.
echo Platform:          Windows
echo Toolset:           %TOOLSET%
echo Install directory: %INSTALL_PREFIX%
echo Build directory:   %BUILD_DIR%
echo.

REM ================================================
REM Create directories
REM ================================================
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%INSTALL_PREFIX%" mkdir "%INSTALL_PREFIX%"

REM ================================================
REM Determine configurations to build
REM ================================================
if /i "%TOOLSET%"=="msvc-asan" (
    if defined CONFIG (
        set CONFIGURATIONS=%CONFIG%
    ) else (
        set CONFIGURATIONS=Asan
    )
) else (
    if defined CONFIG (
        set CONFIGURATIONS=%CONFIG%
    ) else (
        set CONFIGURATIONS=Debug Release
    )
)

REM ================================================
REM Configure based on toolset
REM ================================================
echo Configuring dependencies...
echo Configurations: %CONFIGURATIONS%
echo.

if /i "%TOOLSET%"=="msvc-asan" (
    cmake -S "%DEP_DIR%" -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%" ^
        -DCMAKE_CONFIGURATION_TYPES="Asan;Release;Debug" ^
        -DCMAKE_CXX_FLAGS="/EHsc /MD /Zi /fsanitize=address" ^
        -DCMAKE_C_FLAGS="/fsanitize=address" ^
        -DCMAKE_EXE_LINKER_FLAGS="/INFERASANLIBS" ^
        -DCMAKE_SHARED_LINKER_FLAGS="/INFERASANLIBS" ^
        -DBUILD_EIGEN3=ON ^
        -DBUILD_CATCH2=ON ^
        -DBUILD_ABSEIL=ON ^
        -DBUILD_CERES=ON ^
        -DBUILD_SOPHUS=ON ^
        -DBUILD_ARGS=ON ^
        -DBUILD_MINIZ=ON ^
        -DBUILD_TINYEXR=ON ^
        -DBUILD_SIMDMATH=ON ^
        -DBUILD_GRIDSEARCH=ON ^
        -DDEPS_VERBOSE=OFF ^
        !CMAKE_EXTRA_ARGS!
) else if /i "%TOOLSET%"=="clangcl" (
    cmake -S "%DEP_DIR%" -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%" ^
        -DCMAKE_CONFIGURATION_TYPES="Debug;Release" ^
        -DCMAKE_C_COMPILER=clang-cl ^
        -DCMAKE_CXX_COMPILER=clang-cl ^
        -DBUILD_EIGEN3=ON ^
        -DBUILD_CATCH2=ON ^
        -DBUILD_ABSEIL=ON ^
        -DBUILD_CERES=ON ^
        -DBUILD_SOPHUS=ON ^
        -DBUILD_ARGS=ON ^
        -DBUILD_MINIZ=ON ^
        -DBUILD_TINYEXR=ON ^
        -DBUILD_SIMDMATH=ON ^
        -DBUILD_GRIDSEARCH=ON ^
        -DDEPS_VERBOSE=OFF ^
        !CMAKE_EXTRA_ARGS!
) else (
    cmake -S "%DEP_DIR%" -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%" ^
        -DCMAKE_CONFIGURATION_TYPES="Debug;Release" ^
        -DBUILD_EIGEN3=ON ^
        -DBUILD_CATCH2=ON ^
        -DBUILD_ABSEIL=ON ^
        -DBUILD_CERES=ON ^
        -DBUILD_SOPHUS=ON ^
        -DBUILD_ARGS=ON ^
        -DBUILD_MINIZ=ON ^
        -DBUILD_TINYEXR=ON ^
        -DBUILD_SIMDMATH=ON ^
        -DBUILD_GRIDSEARCH=ON ^
        -DDEPS_VERBOSE=OFF ^
        !CMAKE_EXTRA_ARGS!
)

if errorlevel 1 (
    echo ERROR: CMake configure failed
    exit /b 1
)

REM ================================================
REM Build each configuration
REM ================================================
echo.
echo Building dependencies (this may take a while)...

for %%C in (%CONFIGURATIONS%) do (
    echo.
    echo -- Building %%C configuration --
    cmake --build "%BUILD_DIR%" --config %%C --parallel
    if errorlevel 1 (
        echo ERROR: Build failed for %%C configuration
        exit /b 1
    )

    echo -- Installing %%C configuration --
    cmake --install "%BUILD_DIR%" --config %%C
    if errorlevel 1 (
        echo ERROR: Install failed for %%C configuration
        exit /b 1
    )
)

REM ================================================
REM Clean build directory if requested
REM ================================================
if defined CLEAN_BUILD (
    echo.
    echo Cleaning up build directory: %BUILD_DIR%
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
        echo Build directory removed.
    )
)

echo.
echo ==============================================
echo Dependencies installed successfully!
echo ==============================================
echo.
echo To use these dependencies, configure QuaternionICP with:
echo   cmake --preset windows-%TOOLSET%
echo.
goto :EOF

REM ================================================
REM Help message
REM ================================================
:show_help
echo QuaternionICP Dependencies Builder
echo.
echo Usage: dependencies.bat [TOOLSET] [OPTIONS] [CMAKE_ARGS...]
echo.
echo Toolsets:
echo   msvc              Build with MSVC (default)
echo   clangcl           Build with Clang-CL
echo   msvc-asan         Build with MSVC AddressSanitizer
echo.
echo Options:
echo   --help, -h        Show this help message
echo   --config CONFIG   Build only specified config (Debug, Release, or Asan)
echo                     Default: Debug + Release (or Asan for msvc-asan)
echo   --clean           Delete build directory after install
echo   --ci              CI mode: disable SIMDMath and GridSearch
echo.
echo Optional CMake flags (pass directly):
echo   -DUSE_MKL=ON          Enable Intel MKL for LAPACK/BLAS
echo   -DUSE_TBB=ON          Enable Intel TBB for parallel algorithms
echo   -DUSE_CUDA=ON         Enable CUDA support for GPU acceleration
echo   -DUSE_SUITESPARSE=ON  Enable SuiteSparse for sparse Cholesky
echo   -DDEPS_VERBOSE=ON     Show detailed build output
echo   -DBUILD_SHARED_LIBS=ON  Build shared libraries (.dll)
echo.
echo Examples:
echo   dependencies.bat
echo       Build with MSVC, Debug + Release
echo.
echo   dependencies.bat --config Release
echo       Build with MSVC, Release only
echo.
echo   dependencies.bat clangcl --config Debug
echo       Build with Clang-CL, Debug only
echo.
echo   dependencies.bat msvc -DUSE_CUDA=ON
echo       Build with MSVC and CUDA support
echo.
echo   dependencies.bat --clean
echo       Build and remove build directory after install
echo.
exit /b 0
