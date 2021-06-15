@echo off
set script=%cd%/bin/reporting.py
set python=%cd%/bin/Python/python.exe
call %python% %script% --fpath c:/00_copy/_analysis_FOCUSSW/reporting_v02/projects/AsellusIT/ 
pause