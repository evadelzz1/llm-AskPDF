# llm-AskPDF

Cloning the Repository

    git clone https://github.com/evadelzz1/llm-AskPDF.git

Setting up a Virtual Environment

    cd ./llm-AskPDF

    pyenv versions

    pyenv local 3.11.6

    pyenv versions

    echo '.env'  >> .gitignore
    echo '.venv' >> .gitignore

    echo 'OPENAI_API_KEY=sk-9jz....' >> .env

    ls -la

    python -m venv .venv

    source .venv/bin/activate

    python -V

Install the required dependencies

    pip list
    
    pip install -r requirements.txt
    
    pip freeze | tee requirements.txt.detail

Running the Application

    python -m streamlit run app.py

Deactivate the virtual environment

    deactivate
