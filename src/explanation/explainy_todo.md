

# Test
```python
from explainy.explanations import PermutationExplanation

explainer = PermutationExplanation(model=model, number_of_features=10)
explainer.fit(features=X, target=y)

for sample_index in range(10):
	explainer.explain(sample_index=1)
	explainer.save(sample_name='131')

	explainer.print()
	explainer.plot()
```

Reference:
https://github.com/unit8co/darts


# TODO:
P

- add `examples` folder -> 01-explainy-intro.ipynb
- add `static\images` folder
- let `ExplanationBase` inherit from ABC module, remove not implemented methods with `@abstractmethod`
https://stackoverflow.com/questions/56008847/when-should-one-inherit-from-abc
- remove `sparse` as parameter
- remove `show_score`
- add `number_of_features`
- create new `ExplanationText` class?
- add `_is_explain = False` to `__init__`
- make methods private (`_calculate_importance`)
- rename files -> `permutation_explanation.py`



# Folder structure:


```
explainy (repo_name)

	explainy

		dataprocessing
		explanations
				Permutation
				Shapley
				Surrogate
				Counterfactual
				ExplanationBase
				ExplanationMixin
				ExplanationText
		utils
		models (?)
		tests
		logging.py

	examples
	static
	docs
	requirements

```

# Docs


in `config.py`:


add
```
html_theme = 'pydata_sphinx_theme'
html_logo = "static/darts-logo-trim.png"

html_theme_options = {
  "github_url": "https://github.com/unit8co/darts",
  "twitter_url": "https://twitter.com/unit8co",
  "search_bar_position": "navbar",
}

```