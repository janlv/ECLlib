setlocal enabledelayedexpansion

REM Extract package name from pyproject.toml (assumes format: name = "your_package")
for /f "tokens=2 delims== " %%A in ('findstr /b "name =" pyproject.toml') do (
    set name_raw=%%A
    goto :found
)

:found
REM Clean up the extracted name
set PACKAGE_NAME=!name_raw:"=!
set PACKAGE_NAME=!PACKAGE_NAME: =!
set VENV_DIR=.venv_!PACKAGE_NAME!

echo 🔧 Creating virtual environment in !VENV_DIR! ...
python -m venv "!VENV_DIR!"

echo 🔌 Activating virtual environment ...
call "!VENV_DIR!\Scripts\activate.bat"

echo 📦 Installing local project ...
pip install .

if exist requirements.txt (
    echo ➕ Installing additional dependencies from requirements.txt ...
    pip install -r requirements.txt
) else (
    echo ℹ️ No requirements.txt found – skipping extra dependencies.
)

echo.
echo ✅ Done! Virtual environment created in !VENV_DIR!
echo 💡 To activate it later, run:
echo     call !VENV_DIR!\Scripts\activate.bat
echo 💡 To deactivate the environment, just run:
echo     deactivate
