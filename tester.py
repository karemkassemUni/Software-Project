import os
import subprocess
import sys

# Paths to project executables and scripts
EXECUTABLE = "./symnmf"  # Path to your compiled symnmf executable
PYTHON_SYMNMF = "python3 symnmf.py"  # Path to your symnmf Python script
PYTHON_ANALYSIS = "python3 analysis.py"  # Path to your analysis script
INPUT_FILE = "input.txt"  # Sample input file

# Sample input data for testing
sample_data = """1.0, 2.0\n3.0, 4.0\n5.0, 6.0\n"""


def create_input_file(filename):
    """Creates a sample input file for testing."""
    with open(filename, "w") as f:
        f.write(sample_data)


def run_command(command):
    """Runs a shell command and captures its output."""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        print(f"Command: {command}")
        print("Output:")
        print(result.stdout)
        if result.returncode != 0:
            print("Error:")
            print(result.stderr)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"Failed to execute command: {command}")
        print(e)
        return 1, "", str(e)


def test_executable():
    """Test the symnmf C executable with various goals."""
    print("Testing C executable (symnmf)...")
    commands = [
        f"{EXECUTABLE} sym {INPUT_FILE}",
        f"{EXECUTABLE} ddg {INPUT_FILE}",
        f"{EXECUTABLE} norm {INPUT_FILE}"
    ]
    for command in commands:
        run_command(command)


def test_python_symnmf():
    """Test the symnmf Python script with various goals."""
    print("Testing Python script (symnmf.py)...")
    commands = [
        f"{PYTHON_SYMNMF} 2 sym {INPUT_FILE}",
        f"{PYTHON_SYMNMF} 2 ddg {INPUT_FILE}",
        f"{PYTHON_SYMNMF} 2 norm {INPUT_FILE}",
        f"{PYTHON_SYMNMF} 2 symnmf {INPUT_FILE}"
    ]
    for command in commands:
        run_command(command)


def test_analysis():
    """Test the analysis script comparing SymNMF and K-means."""
    print("Testing analysis script (analysis.py)...")
    command = f"{PYTHON_ANALYSIS} 2 {INPUT_FILE}"
    run_command(command)


def main():
    print("Creating test input file...")
    create_input_file(INPUT_FILE)

    print("\nRunning tests...\n")

    # Test each component
    test_executable()
    test_python_symnmf()
    test_analysis()

    # Clean up
    # print("\nCleaning up test files...")
    # if os.path.exists(INPUT_FILE):
    #     os.remove(INPUT_FILE)
    print("Tests completed.")


if __name__ == "__main__":
    main()
