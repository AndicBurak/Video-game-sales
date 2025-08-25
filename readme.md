# Video Game Sales – ML Project

> This project analyzes global sales of physical video game copies and builds machine learning models to predict whether a game becomes a success (>1 million sold).  
> Through EDA and classification models (Random Forest, XGBoost), I show how class imbalance affects performance and improve predictions by rebalancing and threshold tuning.  
> I also used feature importance analysis to identify which factors matter most for predicting success.

## Why physical copies?
The dataset only includes physical retail sales. Digital distribution (Steam/PS Store/Xbox/eShop/mobile) is **not** included.  
This explains why the overall sales appear to decline after ~2010: the market shifted to digital, not because games became less popular.

## Data & preparation
- Dataset: *Video Game Sales* (Kaggle).  
- Cleaning: I kept the columns `Name, Platform, Year_of_Release, Genre, Publisher, NA/EU/JP/Other/Global_Sales` and removed rows with missing `Year_of_Release` or `Publisher`.  
- Target: `Success = 1` if `Global_Sales > 1.0`, otherwise `0`.

## EDA – highlights
- **Genres**: Action and Sports are the most common.  
- **Platforms**: PS2, X360, and PS3 generated the most total sales.  
- **Trend**: Global sales of **physical copies** peaked around 2008–2010 and declined afterwards.

## Models & results
I tested two models:
- **Random Forest**  
- **XGBoost**

Both models achieved high overall accuracy (~86–89%), but weaker F1-score for the **Success class (1)** (~0.37).  
This shows the challenge of **imbalanced data** (far fewer successes than non-successes), rather than the choice of model.  

When I balanced the data with class weights and tuned the threshold, the model’s ability to detect successes improved.  
F1 for the success class increased from **0.37 → 0.49**, showing that the model now captures more successes, even if it sometimes misclassifies.

## Feature importance
Using the XGBoost model, I analyzed which features were most important:  
- **Year of Release** and **Genre** were the most decisive predictors of success.  
- **Platform** and **Publisher** had less influence.  

This makes sense since some time periods and genres historically had a much higher chance of producing best-sellers.

## Model comparison

| Model                      | Accuracy | Precision (Success) | Recall (Success) | F1 (Success) |
|-----------------------------|----------|---------------------|------------------|--------------|
| Random Forest              | 0.86     | 0.40                | 0.35             | 0.37         |
| XGBoost (default, thr=0.5) | 0.89     | 0.58                | 0.27             | 0.37         |
| XGBoost (bal + thr=0.66)   | 0.85     | 0.41                | 0.61             | 0.49         |

## Conclusion
- Accuracy looks good but is misleading due to imbalance.  
- Success prediction improved when using class weights + threshold tuning.  
- Feature importance shows that release year and genre are the strongest signals for predicting success.  
- The decline in overall physical sales is explained by the shift toward **digital distribution and subscription models**, not a drop in gaming popularity.

## Future work
- Include **digital** sales or balance the dataset (e.g. class weights/SMOTE).  
- Try additional models and threshold tuning for better recall on successes.  
- Add a simple app (e.g. Streamlit) to make predictions interactive.  
- Use feature importance insights to investigate which genres/years could be “hidden gems”.

## Reproduce
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook

projekt/
├─ gamesales.ipynb
├─ games.csv
├─ ps4.csv
├─ xbox.csv
├─ README.md
├─ requirements.txt
└─ .gitignore
