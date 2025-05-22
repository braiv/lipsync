# Braiv Lipsync Project Setup Guide

## Prerequisites

- Windows 10 or later
- Python 3.8 or later
- Git
- Visual Studio Build Tools (for some Python packages)

## Installation Steps

1. **Install Python**
   - Download and install Python from [python.org](https://www.python.org/downloads/)
   - During installation, make sure to check "Add Python to PATH"
   - Verify installation by opening Command Prompt and running:
     ```
     python --version
     ```

2. **Install Visual Studio Build Tools**
   - Download Visual Studio Build Tools from [Microsoft's website](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select "Desktop development with C++"

3. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/lipsync.git
   cd lipsync
   ```

4. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `api/` - API endpoints and related code
- `facefusion/` - Core face fusion functionality
- `scripts/` - Utility scripts
- `service-config/` - Service configuration files
- `tests/` - Test files

## Configuration

1. Copy the example configuration file:
   ```bash
   copy facefusion.ini.example facefusion.ini
   ```

2. Edit `facefusion.ini` with your desired settings

## Running the Application

1. Activate the virtual environment (if not already activated):
   ```bash
   .\venv\Scripts\activate
   ```

2. Start the application:
   ```bash
   python facefusion.py run
   ```

## Common Issues and Solutions

1. **Missing DLL errors**
   - Ensure Visual Studio Build Tools are properly installed
   - Try reinstalling the problematic package:
     ```bash
     pip uninstall package-name
     pip install package-name
     ```

2. **Python path issues**
   - Verify Python is in your system PATH
   - Try using the full path to Python in your commands

## Support

For issues and support, please:
1. Check the existing issues in the repository
2. Create a new issue with detailed information about your problem

## License

This project is licensed under the terms included in the LICENSE.md file.
