Titanic EDA — Who Survived and Why?
> Exploratory data analysis on the Titanic passenger dataset to uncover the key factors that determined survival.
---
Overview
This project performs a full exploratory data analysis on the Titanic dataset (891 passengers, 12 features). The goal is to understand which factors — class, sex, age, fare, family size — most strongly predicted survival, and to prepare a clean, feature-engineered dataset for downstream ML modelling.
---
Key Findings
Finding	Detail
Sex was the strongest predictor	Women survived at 74% vs men at 19%
Class had a near-linear effect	1st class: 63% survival — 3rd class: 24%
Children had better odds	Passengers under 12 survived at ~58%
Travelling alone was a disadvantage	Solo passengers survived at 30% vs 50% for small families
Higher fare = higher survival	Strong positive correlation (r = 0.26)
---
Project Structure
```
titanic-eda/
│
├── titanic_eda.ipynb      
├── titanic.csv            
├── requirements.txt       
└── README.md
```
---
Notebook Walkthrough
The analysis is broken into 6 sections:
Setup & data loading — import libraries, load CSV, first look
Data understanding — shape, dtypes, `.describe()`, missing value audit
Missing data handling — median imputation for Age (group-based by Title), mode for Embarked, drop Cabin
Univariate analysis — distributions of Age, Fare, Pclass, Sex, Embarked
Bivariate analysis — survival rate broken down by each feature; correlation heatmap
Feature engineering — FamilySize, IsAlone, AgeGroup, Title extraction
---
Visualisations
Plot	Insight
Survival by sex (bar)	Women survived at 3.7× the rate of men
Survival by class (bar)	Near-linear drop from 1st → 3rd class
Age distribution by survival (box)	Survivors skewed slightly younger
Fare vs survival (KDE)	Higher fares strongly associated with survival
Correlation heatmap	Fare and Pclass are most correlated with survival
---
Missing Data Strategy
Column	Missing	Strategy	Reason
Age	177 (20%)	Group median by Title	Preserves age patterns across passenger types
Cabin	687 (77%)	Dropped + `HasCabin` binary	Too sparse; missingness itself is informative
Embarked	2 (0.2%)	Mode fill	Negligible, safe to fill with most common port
---
Setup & Usage
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/titanic-eda.git
cd titanic-eda

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook titanic_eda.ipynb
```
requirements.txt
```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
scikit-learn==1.3.0
```
---
What I Learned
How to audit and handle missing data systematically rather than arbitrarily
Why group-based imputation (by Title) outperforms global median for the Age column
How to use `pd.cut()` for binning continuous variables into meaningful categories
How bivariate analysis reveals interactions that univariate analysis hides (e.g. sex × class interaction)
---
Dataset
Titanic dataset via Kaggle. Originally from the Encyclopedia Titanica.
