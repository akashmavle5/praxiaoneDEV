#!/bin/bash
echo "Building package..."
python setup.py sdist bdist_wheel
echo "Build complete."
