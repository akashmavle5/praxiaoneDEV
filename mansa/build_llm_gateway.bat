@echo off
echo Building ai_llm_gateway package...

set PACKAGE=ai_llm_gateway

rmdir /s /q %PACKAGE%
rmdir /s /q dist
rmdir /s /q build

mkdir %PACKAGE%

echo from setuptools import setup, find_packages > %PACKAGE%\setup.py
echo setup(name="ai_llm_gateway", version="1.0.0", packages=find_packages()) >> %PACKAGE%\setup.py

cd %PACKAGE%

pip install wheel
python setup.py sdist bdist_wheel
pip install dist\*.whl

echo DONE
pause
