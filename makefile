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


black_long_string:
	black src -l 80 --experimental-string-processing


setup:
	conda env create -f environment.yml

env:
	conda activate explanation_env

export_env:
	conda env export > environment.yml 

coverage:
	python -m coverage run -m pytest	

report:
	python -m coverage report -m