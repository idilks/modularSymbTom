@echo off
echo Copying files to Dartmouth HPC...

REM Set destination path
set DEST_DIR=\\dartfs.dartmouth.edu\rc\lab\F\FranklandS\tom

REM Check if destination is accessible
if not exist "%DEST_DIR%" (
    echo Error: Cannot access %DEST_DIR%
    echo Make sure you're connected to the Dartmouth network and have proper permissions
    pause
    exit /b 1
)

REM Copy files one by one
echo Copying Python files...
copy "*.py" "%DEST_DIR%\" 2>nul
if %ERRORLEVEL% EQU 0 echo Python files copied successfully

@REM echo Copying Markdown files...
@REM copy "*.md" "%DEST_DIR%\" 2>nul
@REM if %ERRORLEVEL% EQU 0 echo Markdown files copied successfully

echo Copying text files...
copy "*.txt" "%DEST_DIR%\" 2>nul
if %ERRORLEVEL% EQU 0 echo Text files copied successfully

@REM echo Copying PDF files...
@REM copy "*.pdf" "%DEST_DIR%\" 2>nul
@REM if %ERRORLEVEL% EQU 0 echo PDF files copied successfully


echo Copying shell files...
copy "*.sh" "%DEST_DIR%\" 2>nul
if %ERRORLEVEL% EQU 0 echo shell files copied successfully

echo Copying codebase (forcing overwrite)...
robocopy "codebase" "%DEST_DIR%\codebase" /MIR /R:0 /NP /NDL /XD ".ipynb_checkpoints" "__pycache__" "results" >nul 2>&1
if %ERRORLEVEL% LEQ 3 echo codebase folder synced successfully

echo Copying behavioral experiments...
robocopy "behavioral" "%DEST_DIR%\behavioral" /MIR /R:0 /NP /NDL /XD ".ipynb_checkpoints" "__pycache__" "results" >nul 2>&1
if %ERRORLEVEL% LEQ 3 echo behavioral folder synced successfully



echo Copying bat files...
copy "*.bat" "%DEST_DIR%\" 2>nul
if %ERRORLEVEL% EQU 0 echo bat files copied successfully

echo Copy operation completed!


for /f "usebackq delims=" %%x in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set TIMESTAMP=%%x
echo Finished at %TIMESTAMP%




pause