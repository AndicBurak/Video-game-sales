# Video Game Sales – ML-projekt

I det här projektet analyserar jag global försäljning av **fysiska spelkopior** och tränar modeller för att förutsäga om ett spel blir en succé (> 1 miljon sålda). Fokus är på en enkel, tydlig process med minimalt med kod.

## Varför fysiska kopior?
Datasetet innehåller endast fysisk retail-försäljning. Digital distribution (Steam/PS Store/Xbox/eShop/mobil) ingår **inte**. Det förklarar varför den totala försäljningen sjunker efter ~2010: marknaden flyttar till digitalt, inte att spel blivit mindre populära.

## Data & förberedelser
- Data: *Video Game Sales* (Kaggle).  
- Städning: Jag behöll kolumnerna `Name, Platform, Year_of_Release, Genre, Publisher, NA/EU/JP/Other/Global_Sales` och slängde rader med saknat `Year_of_Release`/`Publisher`.  
- Målvärde: `Success = 1` om `Global_Sales > 1.0`, annars `0`.

## EDA – kort
- **Genrer**: Action och Sports är vanligast.
- **Plattformar**: PS2, X360 och PS3 står för mest totalsålt.
- **Trend**: Global försäljning av **fysiska kopior** toppar runt 2008–2010 och minskar därefter.

## Modeller & resultat
Jag testade två modeller:
- **Random Forest**  
- **XGBoost**

Båda gav hög total accuracy (~86–89%), men sämre f1-score för klassen **Succé (1)** (~0.37). Slutsats: problemet är **obalanserad data** (färre succéer än icke-succéer), inte modellvalet i sig.

## Slutsats
- Accuracy ser bra ut men luras av klass-obalans.  
- Modellerna är mycket bättre på att känna igen icke-succéer än succéer.  
- Nedgången i total försäljning beror sannolikt på skiftet till **digital distribution och abonnemang**, inte minskat spelintresse.

## Vidare arbete
- Inkludera **digital** försäljning eller balansera data (t.ex. class weights/SMOTE).  
- Testa fler modeller och threshold-tuning för bättre recall på succéer.  
- Lägg till enkel app (t.ex. Streamlit) om jag vill visa prediktioner interaktivt.

## Reproducera
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
