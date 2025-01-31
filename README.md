# Mixed-Integer Linear Parlays, An Optimized Sports Betting Approach

## Overview of the Project
This project used the NBA_api to gather data on the outcomes of games from the 2020-2024 NBA seasons. The data was used create a characteristic feature set that trained a supervised machine learning model in order predict the outcome a home team winning any given game in the 2024-2025 season. Using the predictions, a betting strategy was developed to maximize the expected value of profit that was solved using a mixed-integer linear programming model.

## Installation
To install the necessary packages to run this project, use the following command:
```bash
pip install nba_api numpy pandas matplotlib sklearn pulp json difflib requests time
```

## Usage
To create the data sets for the project, run the following command:
```bash
win_loss_data_prep.py
```
This tends to take several hours due to api call limits and the vast amount of data being requested. However, the data is already saved in the csv_data_file folder so this step can be skipped. 

Specifically, the following files in the csv_data_files folder is are used to train the machine learning model and create the feature set for the optimization strategy: 
- 2020-25_scheduleFrame.csv
- 2020-25_gameLogFrame.csv
- 2020-25_nbaHomeWinLossModelDataSet.csv

The other following files in the csv_data_files folder were created to ensure functionality of the code but are not used in the final project.
- 2024-25_scheduleFrame.csv
- 2024-25_gameLogFrame.csv
- 2024-25_nbaHomeWinLossModelDataSet.csv

To train the machine learning model and evaluate the model's performance, run the following command after specifiying the model type in the main function:
```bash
win_loss_data_model.py
```
In addition to the previously mentioned data files, the following file is required to generate the moneyline multipliers for the optimization strategy:
- betting_odds.csv

The result of the modeling script create 4 plots that show the Reciever Operating Characteristic (ROC) curve and the time series of correct predictions for the model with predicted probabilities of home team winning on the y-axis. These two plots are created for the training and validation data sets.

Additionally, a csv file labled:
- <model_type>_optimal_wagers.csv
Demonstrates the optimal wagers as determined by the optimization strategy.

## Context for the Project
This project was completed as a final project for ASEN 6519: Optimization Algorithms and Applications at the University of Colorado Boulder. In addition to the code base, a final paper was written and a presenation was given to the class detailing the data collection, feature extraction, model training, and optimization strategy.

## Solution Approach
### Data Collection
The NBA_api (https://github.com/swar/nba_api.git) provides  API client allowing for extraction of game by game data for completed games in the NBA season dating back several years. Using two endpoints, cumestatsteamgames, cumestatsteam, data was collected for the 2020-2025 seasons. This data detailed the outcomes of games and the team statistics for each game. This data was cleaned and processed using the pandas library in Python to create a feature set for the machine learning model.

### Feature Extraction
The specific feature set used in this project included the following features:
- Home Team Win Percentage
- Away Team Win Percentage
- Total Win Percentage
- Offensive Efficiency
- Rolling Offensive Efficiency
- Rolling Scoring Margin
- Number of Rest Days
These features were reccomended by a tutorial available inside the nba_api repository.

### Model Training
This project sought to compare two different supervised machine learning models to predict the outcome of a game. The first model was a logistic regression model and the second was a a Multi-Layer Perceptron (MLP) model. The models were trained on the same feature set using a 75/25 train/test split. The models were validated against results from the 2024-2025 season. Importantly, in addition to the accuracy of the models, the models were evaluated on their ability to predict probabilities of a home team win. This was important for the optimization strategy.

### Optimization Strategy
The optimization strategy was to maximize the expected value of profit from betting on NBA games. The optimization problem was formulated as a mixed-integer linear programming problem. The decision variables were wether to bet on a given game and if so the amount of money bet on that game. The objective function was the expected value of profit from betting on the games calculated using the following formula:
\[
    \mathbb{E}[b_i] = \hat{p_i} \cdot (b_i \cdot \mathcal{R}_i) - (1-\hat{p_i}) \cdot b_i
\]
Where:
- \(\hat{p_i}\) is the predicted probability of the home team winning the game \(i\)
- \(b_i\) is the amount of money bet on the game \(i\)
- \(\mathcal{R}_i\) is the moneyline multiplier for home team winning the game \(i\)
The constraints for the optimization problem were as follows:
- The total amount of money bet could not exceed a set budget
- The total number of bets could not exceed a set number of bets
- The amount of money bet on a single game could not exceed a set maximum bet
- The amount of money bet on a single game must be non-negative
- The decision variable for betting on a game must be binary

## Results
The results of the project were mixed (no-pun intended). The logistic regression model performed better than the MLP model in terms of overall accuracy and ability to differentiate between the positive and zero class. This implies it was able to more effectively identify home wins and differentiate between a home team win and los. Additionally, the logistic regression produced much less extreme probabities that aided in the subsequent expected value calculation. On the other hand, the MLP had a much larger cross-entropy loss indicating that it was not nearly confident in its predictions. While the MLP has a higher recall value, implying it was able to identify home losses more effectively, this was liekly due to the model simply classifiying more games in the zero class and it is not indictive of a better model. Overall the results comparing these models are tabulated in the attached paper in Tables 2,3,4, and 5 with assosiated Figures 2 and 3. 

The optimization strategy was able to produce a betting strategy that when given $1000 over four seperate days produced a loss of $353.33. While this is not terrible, the optimizer typically chose to wager very large amounts of games that had a moderate probability of winning was extremely larger profit multipliers. In other words, the optimizer liked to go big when there was a chance to win a lot of money. This makes sense as the objective function did not incentivise taking safer wagers that had higher probabilities of winning but lower profit multipliers. This is an expected limitation of the formulation and often considered gambling bias. Moderately probably events that have the potenial to win a lot of money are often overvalued by gamblers and this is reflected in the optimizer's strategy.

## Contributions
The data collection, feature extraction, logistic model training were heavily guided by a tutorial inside the NBA_api documentation labeled "Home Team Win-Loss Modeling". While the tutorial was closesly followed, the code was adapted to fit the specific needs of this project. The MLP model training and optimization strategy were developed independently. The optimization strategy was developed with the help of the PuLP library in Python. The final paper and presentation were completed independently.