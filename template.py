import os
from pathlib import Path
import logging

# Configure logging to see the script in action
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "datascience"

# Expanded to reflect industry standards
list_of_files = [
    # CI/CD Pipeline
    ".github/workflows/ci.yaml", 
    
    # Core Package
    f"src/{project_name}/__init__.py",
    
    # Custom Logging and Exceptions (Crucial for production debugging)
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    
    # ML Components (The actual stages of the project)
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    
    # Utilities
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    
    # Configuration Management
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    
    # Execution Pipelines
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    
    # Data Entities (Type hinting and structure)
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    
    # Constants
    f"src/{project_name}/constants/__init__.py",
    
    # Config Files
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    
    # Application & Entry Points
    "main.py",
    "app.py",
    
    # Docker & Deployment
    "Dockerfile",
    ".dockerignore",
    
    # Packaging & Dependencies
    "setup.py",
    "requirements.txt",
    
    # Research & Notebooks
    "research/01_data_exploration.ipynb",
    "research/02_model_experiments.ipynb",
    
    # Testing (Mandatory in real-world projects)
    "tests/__init__.py",
    "tests/test_common.py",
    
    # Local Data Storage (Ignored by Git later, but structure is needed)
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    
    # Web Templates (If serving via Flask/FastAPI)
    "templates/index.html",
    
    # Git Configuration
    ".gitignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")