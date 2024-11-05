import os
import subprocess
import sys

def run_django_server():
    # Determine the virtual environment's activate script path
    if os.name == 'nt':  # Windows
        activate_script = ".venv\\Scripts\\activate"  # Path to the activate script
        python_executable = ".venv\\Scripts\\python.exe"   # Path to the Python executable
    else:  # Unix/Mac
        activate_script = ".venv/bin/activate"      # Path to the activate script
        python_executable = ".venv/bin/python"      # Path to the Python executable

    # Check if the virtual environment is set up correctly
    if not os.path.exists(activate_script) or not os.path.exists(python_executable):
        print("Error: Virtual environment not found or not set up properly.")
        sys.exit(1)

    # Get the PORT from environment variables, default to 8000 if not set
    port = os.getenv("PORT", "8000")

    # Prepare the command to run the Django server
    command = [python_executable, "mysite/manage.py", "runserver", port]

    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start the Django server: {e}")

if __name__ == "__main__":
    run_django_server()
