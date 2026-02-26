# 🚀 Praxiaone Enterprise AI Cost Controller

A full-stack, enterprise-grade AI routing and cost management engine. This middleware layer dynamically calculates Return on Investment (ROI), enforces strict financial budgets, scrubs sensitive data (DLP), and routes prompts across multiple AI models (OpenAI, Anthropic, Gemini, Grok, and local Ollama) using real-time streaming.

---

## 🛠️ How to Run the Application

There are two ways to launch this application on your local machine. You can either run the pre-built Docker container (Fastest), or build the entire architecture from scratch using the bootstrap script (For Developers).

### Option 1: Run the Pre-Built Docker Container (Recommended)
This method requires zero configuration. It runs a completely isolated Linux environment containing the app, database, and all dependencies.

**Prerequisites:** You must have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### tar file link
https://drive.google.com/file/d/17sunhQ4pk5Kf0xuKAKq8cL3hZGLGplEH/view?usp=drive_link

## download the tar file and follow the following steps

1. **Load the Image:**
   Open your terminal in the folder containing the `.tar` file and run:
   
```bash
   docker load -i praxiaone-ai.tar
   ```

### Launch the Server:

``` Bash
docker run -p 8000:8000 praxiaone-ai-controller
```

Access the Dashboard:

Open your web browser and go to: http://localhost:8000/dashboard/

### Option 2: Build From Source via Bootstrap Script
If you want to view the source code, build the Python .whl package natively, and run it on your own machine's Python environment, use the self-extracting bootstrap script.

Prerequisites: Python 3.10+ and a Bash terminal (Linux, macOS, or Git Bash on Windows).

Make the script executable (Linux/macOS/Git Bash):

``` Bash
chmod +x build_praxiaone.sh
```
Run the God Script:
This will automatically generate the folders, write the code, compile the Python Wheel (.whl), install dependencies into a virtual environment, and configure the SQLite database.

```Bash
./build_praxiaone.sh
```
Start the Server:
Once the script finishes successfully, activate the environment and start Django:

```Bash
source venv/Scripts/activate  # (Use venv/bin/activate on Mac/Linux)
python manage.py runserver
```
Access the Dashboard:
Open your web browser and go to: http://localhost:8000/dashboard/

🔐 Default Credentials
The bootstrap script automatically generates a superuser for you to test the administrative features (like setting user budgets and viewing telemetry).

Username: omkar

Password: admin

(Note: You can use these credentials to log into http://localhost:8000/admin/)