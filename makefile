.DEFAULT: env


init:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

freeze:
	pip freeze > requirements.txt

flake8:
	python -m flake8 src

black_diff:
	black src --color --diff

black:
	isort src/
	black src


type_checking:
	mypy src/model/ModelClass.py
	mypy src/model/LoggerClass.py


setup:
	conda env create -f environment.yml

env:
	conda activate explanation_env

export_env:
	conda env export > environment.yml 
