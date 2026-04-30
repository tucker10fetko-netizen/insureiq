# InsureIQ — ML Insurance Premium Estimator

A full-stack machine learning web app that estimates health, home, auto, and life insurance premiums using trained regression models.

## Stack
Python · Flask · scikit-learn · TensorFlow · HTML/CSS/JS

## Setup
1. Clone the repo
2. Install dependencies: `pip install flask scikit-learn tensorflow numpy pandas`
3. Download the trained models from [Google Drive](https://drive.google.com/drive/folders/1ekxBPjNe32-vM5yW9pRGXOu1kh1FcNur?usp=drive_link) and place them in the `/models` folder
4. Run: `python InsuranceApp.py`
5. Open `http://127.0.0.1:5000`

## Models
| Insurance | Model | R² |
|-----------|-------|-----|
| Health | Linear Regression | 0.78 |
| Home | Random Forest | 0.53 |
| Auto | Random Forest | 0.98 |
| Life | Random Forest | 0.72 |
