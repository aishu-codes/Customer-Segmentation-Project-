# Customer Segmentation Studio

Beginner-friendly customer analytics dashboard using Python, Streamlit, pandas, scikit-learn, and Matplotlib.

## Features
- K-Means clustering for customer segmentation
- Interactive number-of-segments control
- Selectable clustering features
- Upload a CSV dataset
- Filter customers by gender, preferred channel, and segment
- KPI cards and income-vs-spending scatter plot
- Segment-size chart and profile table
- Simple marketing recommendations based on each segment's profile
- Download the filtered and segmented data as CSV

## Run on Windows
1. Extract `Customer_Segmentation_Project.zip`.
2. Open the extracted folder in VS Code.
3. Open Terminal: **Terminal → New Terminal**.
4. Create a virtual environment (recommended):
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run this for the current terminal only, then activate again:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
6. Install libraries:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Start the dashboard:
   ```powershell
   python -m streamlit run app.py
   ```
8. A browser tab should open with the dashboard. If it does not, copy the local URL printed in Terminal (usually `http://localhost:8501`) into your browser.

## CSV columns
Required columns:
- `Age`
- `AnnualIncome_k` (annual income in thousands)
- `SpendingScore` (sample behavior score from 1 to 100)

Optional columns:
- `CustomerID`
- `Gender`
- `AnnualVisits`
- `PreferredChannel`

Example: see `data/customers.csv`.

## How it works
1. StandardScaler normalizes selected numeric features so their scales do not dominate the model.
2. K-Means groups customers into the chosen number of clusters.
3. Charts and profile statistics help interpret each segment.
4. Rule-based suggestions provide example marketing actions.

## Important
- Included customer records are fictional sample data for learning.
- Cluster IDs are arbitrary; profile each group before assigning a business label.
- SpendingScore is a demonstration index, not actual money spent.
- The project is an educational prototype, not a production customer profiling system.
