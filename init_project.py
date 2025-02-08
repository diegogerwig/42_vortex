import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

def get_config():
    home = os.environ['HOME']
    conda_path = f"{home}/sgoinfre/miniforge"
    return {
        'conda_path': conda_path,
        'conda_bin': f"{conda_path}/bin/conda",
        'env_name': "vortex_env",
        'env_path': f"{conda_path}/envs/vortex_env"
    }

def run_command(command, shell=False, capture_output=False):
    try:
        if shell:
            if capture_output:
                return subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            return subprocess.run(command, shell=True, check=True)
        else:
            if capture_output:
                return subprocess.run(command.split(), check=True, capture_output=True, text=True)
            return subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Error executing: {command}")
        return False

def remove_venv(config):
    print('\n🧹 Removing conda environment')
    run_command(f"{config['conda_bin']} env remove -n {config['env_name']} -y", shell=True)
    if os.path.exists(config['env_path']):
        shutil.rmtree(config['env_path'], ignore_errors=True)

def create_venv(config):
    print('\n🔆 Creating new conda environment')
    return run_command(f"{config['conda_bin']} create -y -n {config['env_name']} python=3.11")

def activate_venv(config):
    print('\n🐍 Activating conda environment')
    activate_command = f". {config['conda_path']}/etc/profile.d/conda.sh && conda activate {config['env_name']}"
    return run_command(activate_command, shell=True)

def install_dependencies(config):
    print('\n🔗 Installing dependencies')
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    return run_command(f"{config['env_path']}/bin/pip install -r requirements.txt")

def run_project(config):
    if not os.path.exists("./src/split.py"):
        print("❌ File not found")
        return False

    python_bin = f"{config['env_path']}/bin/python"
    commands = [
        (f"{python_bin} ./src/split.py", '📂 Splitting dataset\n'),
    ]

    for cmd, msg in commands:
        print(f"\n{msg}")
        if not run_command(cmd):
            return False
    return True

def clean_project():
    print('\n🧹 Starting project cleanup...')
    
    total_deleted = 0
    
    # Files to clean
    files = [
        "./data/processed/data_training.csv",
        "./data/processed/data_validation.csv",
        "./data/processed/data_test.csv",
    ]
    
    # Directories to clean
    dirs = []
    
    # Clean files
    files_deleted = 0
    print("\nCleaning individual files:")
    print("=" * 40)
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Removed: {file}")
            files_deleted += 1
        else:
            print(f"✕ Skipping: {file} (not found)")
    total_deleted += files_deleted

    # Clean directories
    print("\nCleaning directories:")
    print("=" * 40)
    for dir_path in dirs:
        dir_files_deleted = 0
        if os.path.exists(dir_path):
            print(f"\nProcessing directory: {dir_path}")
            for item in os.listdir(dir_path):
                if item != '.gitkeep':
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f"✓ Removed file: {item}")
                        dir_files_deleted += 1
                    elif os.path.isdir(item_path):
                        items_in_subdir = sum(len(files) for _, _, files in os.walk(item_path))
                        shutil.rmtree(item_path)
                        print(f"✓ Removed directory: {item} (containing {items_in_subdir} files)")
                        dir_files_deleted += items_in_subdir
            if dir_files_deleted == 0:
                print("📂 Directory was already clean")
            else:
                print(f"📊 {dir_files_deleted} items removed from {dir_path}")
        else:
            os.makedirs(dir_path, exist_ok=True)
            Path(os.path.join(dir_path, '.gitkeep')).touch()
            print(f"📁 Created directory: {dir_path}")
        total_deleted += dir_files_deleted

    # Print summary
    print("\n✨ Cleanup Summary:")
    print("=" * 40)
    print(f"🗑️  Total items removed: {total_deleted}")
    print(f"📄 Individual files removed: {files_deleted}")
    print(f"📁 Directories processed: {len(dirs)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python init_project.py [-option]")
        print("Options:")
        print("  -init     : Create env & activate env & install dependencies & run project")
        print("  -run      : Activate env & run project")
        print("  -clean    : Remove unnecessary files and directories")
        print("  -remove   : Remove env & Remove unnecessary files and directories")
        return

    config = get_config()
    option = sys.argv[1]

    if option == "-init":
        remove_venv(config)
        clean_project()
        if not create_venv(config):
            return
        if not activate_venv(config):
            return
        if not install_dependencies(config):
            return
        run_project(config)
    elif option == "-run":
        clean_project()
        if not activate_venv(config):
            return
        if not install_dependencies(config):
            return
        run_project(config)
    elif option == "-clean":
        clean_project()
    elif option == "-remove":
        remove_venv(config)
        clean_project()
    else:
        print(f"❌ Invalid option: {option}")

if __name__ == "__main__":
    main()