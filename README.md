# Text Summarizer Pipeline

This is Project made with the help of Groq and TextGrad.

## Running project locally

### Method 1 - Using Nix (For UNIX based systems)

Make sure you have the [Nix package manager](https://nixos.org/download/) and Git installed.

1. Clone the repository

```bash
git clone https://github.com/Natural-Language-Computing/Text-Summarizer-Pipeline.git
cd Text-Summarizer-Pipeline
```

2. Create `.env` file and put your [Groq](https://groq.com) API key in it

```env
GROQ_API_KEY="your API key"
```

3. Enable nix shell

```bash
nix develop --extra-experimental-features "nix-command flakes" --no-pure-eval

# This command will install Python, create a virtual environment, and install dependencies required for the project
# Refer to `flake.nix` for more details.
```

4. Run the project

```bash
streamlit run pipeline.py
```

5. Open the browser and go to [`http://localhost:8501`](http://localhost:8501)

### Method 2 - Traditional Method

Make sure you have the following installed:
- Git
- Python 3.11 or higher

To run the project locally, follow these steps:

1. Clone the repository

```bash
git clone https://github.com/Natural-Language-Computing/Text-Summarizer-Pipeline.git
cd Text-Summarizer-Pipeline
```

2. Create `.env` file and put your [Groq](https://groq.com) API key in it

```env
GROQ_API_KEY="your API key"
```

3. Create a virtual environment

```bash
python3 -m venv venv
```

4. Activate the virtual environment

```bash
# On Windows
venv\Scripts\Activate.ps1

# On Linux or macOS
source venv/bin/activate
```

5. Install the dependencies

```bash
python3 -m pip install -r requirements.txt
```

6. Run the project

```bash
streamlit run pipeline.py
```

7. Open the browser and go to [`http://localhost:8501`](http://localhost:8501)
