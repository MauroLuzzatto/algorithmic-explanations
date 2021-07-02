.DEFAULT: env


init:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

flake8:
	python -m flake8 src

black_diff:
	black src --color --diff

isort:
	src

black:
	black src

setup:
	conda env create -f environment.yml

env:
	conda activate explanation_env

export_env:
	conda env export > environment.yml 
