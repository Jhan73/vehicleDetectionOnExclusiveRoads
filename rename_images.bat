@echo off
setlocal enabledelayedexpansion

rem Define the source directory
set source_dir=./datasets/segmentation/masks
  
rem Define the counter
set /a counter=1

rem Loop through each file in the source directory
for %%f in (%source_dir%\*.jpg) do (
    rem Format the counter to three digits
    set formatted_counter=000!counter!
    set formatted_counter=!formatted_counter:~-4!

    rem Define the new file name
    set new_name=image-!formatted_counter!.jpg

    rem Rename the file
    ren "%%f" "!new_name!"

    rem Increment the counter
    set /a counter+=1
)

endlocal
