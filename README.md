# Real Estate Price Prediction Project

## What This Project Does

This project uses machine learning to estimate the price of a property.

The model uses basic information about a property and its location.

This is only an estimate. It is not a professional real estate appraisal.

## Dataset

This project uses the UCI Real Estate Valuation dataset.

Citation:

Yeh, I. (2018). *Real Estate Valuation* [Dataset]. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C5J30W

License: CC BY 4.0

The model predicts:

- House price per unit area

The dataset uses 6 input features:

1. Transaction date
2. House age
3. Distance to the nearest MRT station
4. Number of nearby convenience stores
5. Latitude
6. Longitude

The `No` column was not used because it is only a row number.

## Data Check

Before training the model, I checked:

- Column names
- Data types
- Missing values
- Minimum and maximum values
- Summary statistics

No missing values were found.

## Feature Choices

I kept all 6 real input features.

I kept transaction date because property prices can change over time.

I kept house age because older and newer homes may have different values.

I kept MRT distance because being close to transportation may affect price.

I kept the number of convenience stores because nearby stores may affect property value.

I kept latitude and longitude because location is very important in real estate.

I removed the `No` column because it is only an ID number.

## Training and Testing

I split the dataset into two parts:

- 331 rows for training
- 83 rows for testing

The model learned from the training data.

The test data was kept separate so I could see how the model worked on data it had never seen before.

## Preprocessing

I used a Scikit-Learn Pipeline.

The pipeline:

1. Fills in missing numbers with the median if needed
2. Scales the numbers
3. Runs the Random Forest model

## Model

I used a Random Forest Regressor.

The model used 300 trees.

I used `random_state=42` so the results can be reproduced.

## Results

The model was tested on data it had not seen before.

Results:

- MAE: 3.93
- RMSE: 5.67
- R²: 0.808

The MAE of 3.93 means the model was off by about 3.93 target units on average.

That is about NT$39,300 per Ping.

The R² score of 0.808 means the model explained about 80.8% of the price differences in the test data.

## Why We Need Test Data

If we only test the model on the same data it learned from, the results can look better than they really are.

The model may remember patterns from the training data.

This is called overfitting.

Testing on separate data gives a better idea of how the model may work on new properties.

## Example Predictions

### Property 1

- House age: 10 years
- MRT distance: 300
- Convenience stores: 8
- Estimated price: 51.58 units
- About NT$515,849 per Ping

### Property 2

- House age: 25 years
- MRT distance: 1200
- Convenience stores: 4
- Estimated price: 25.29 units
- About NT$252,853 per Ping

### Property 3

- House age: 5 years
- MRT distance: 100
- Convenience stores: 10
- Estimated price: 59.63 units
- About NT$596,333 per Ping

## Limits of the Model

This model is only a simple prototype.

It does not know everything that a real estate appraiser would use.

It does not include things like:

- Number of bedrooms
- Number of bathrooms
- Property size
- Building condition
- Renovations
- Recent nearby sales
- Current market conditions

The data also comes from one location and one time period.

Because of that, the model may not work as well in a different city, country, or time period.

That is why the result is an estimate and not an official appraisal.

## How to Run the Project

Install the needed packages with:

python3 -m pip install -r requirements.txt

Run the program with:

python3 real_estate_model.py