# HOBB
A unified cryptography platform for steganography and encryption analysis

### Team Members
---
1. Aadithya Sai G Menon
2. Amala Gopinath
3. Gopika Chandran A J
4. Nadim Naisam

### Instructions
---
Follow these steps to set up and run the project:

#### Clone the Repository

```bash
git clone https://github.com/AadithyaSai/HOBB.git
cd HOBB
```

#### Install Poetry
Ensure you have Poetry installed. You can install it using:

##### Linux, Mac or WSL
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
##### Windows
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

#### Install Dependencies
Use Poetry to install project dependencies:

```bash
poetry install
```

#### Run the FastAPI Application
Use Poetry to directly run HOBB:

```bash
poetry run fastapi dev app/main.py 
```

HOBB should now be accessible at http://127.0.0.1:8000
