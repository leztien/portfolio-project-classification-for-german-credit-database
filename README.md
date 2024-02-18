# EDA and ML model for the German Credit dataset


## Jupyter notebook with EDA:
[exploratory_data_analysis.ipynb](/exploratory_data_analysis.ipynb)


## Dataset description:
The German Credit dataset provides insights into the factors that financial institutions consider when determining the creditworthiness of an applicant. Featuring a mix of numerical and categorical attributes, this dataset presents opportunities for various forms of data analysis, machine learning, and prediction modeling. By understanding the correlations and patterns within this data, one can develop predictive models to determine the likelihood of an approval based on an applicant’s credit, or even spot potential biases in the credit decision-making process.

The dataset has been sourced from Professor Dr. Hans Hofmann of the Universität Hamburg. It comprises 1000 instances with attributes capturing an applicant’s financial behavior, history, and personal details. For instance, it includes attributes such as the status of the applicant’s checking account, credit history, purpose for the loan, and personal information like age and job type. Two versions of the dataset are provided: the original dataset (german.data), which contains a mix of numerical and categorical attributes, and a modified dataset (german.data-numeric), formatted for algorithms that prefer numerical input, wherein categorical variables have been transformed into numerical indicators.


## Information about the dataset:
[info](/data/german.doc)

https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

https://www.interviewquery.com/p/classification-projects

https://github.com/jbrownlee/Datasets/blob/master/german.csv

https://www.kaggle.com/code/janiobachmann/german-credit-analysis-a-risk-perspective

https://www.kaggle.com/code/kabure/predicting-credit-risk-model-pipeline?scriptVersionId=7037624




## TODO:
- EDA
- ML model


## The usual "shell preparations":
```shell
# getting started
$ git init myrepo
$ cd myrepo
$ git branch -M main
$ mkdir data src assets
$ touch README.md requirements.txt
$ echo "# todo" > README.md
$ vim requirements.txt
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ git add README.md requirements.txt
$ git commit -m “my first commit”

# create a new repo on GitHub; generate a token
$ git remote add origin https://github.com/leztien/myrepo.git
$ git pull origin main --rebase  #??
$ git push --set-upstream origin 
$ code .

# routine commits
$ git add README.md requirements.txt
$ git commit -m "comment"
$ git push

# deployment
$ pip freeze > requirements.txt
$ copy the contents of this folder into a new 'deployment' folder and cd into it
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ uvicorn main:app --reload
```

### ML Project checklis:
https://github.com/leztien/handson-ml3-forked/blob/main/ml-project-checklist.md




