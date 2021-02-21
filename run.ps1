$Env:TRAINING_DATA="H:\Portfolio_DS\ML_Framework\input\train_folds.csv"
$Env:TEST_DATA="H:\Portfolio_DS\ML_Framework\input\test.csv"
$Env:MODEL = "extratrees"

$Env:FOLD=0
python -m src.train
$Env:FOLD=1 
python -m src.train
$Env:FOLD=2
python -m src.train
$Env:FOLD=3
python -m src.train
$Env:FOLD=4
python -m src.train

# Prediction

python -m src.predict

[System.Environment]::SetEnvironmentVariable('FOLD', '')
[System.Environment]::SetEnvironmentVariable('TRAINING_DATA', '')
[System.Environment]::SetEnvironmentVariable('TEST_DATA', '')
[System.Environment]::SetEnvironmentVariable('MODEL', '')

 # to get all environement variable
 # Get-ChildItem Env: