# Mixed-Integer Linear Parlays, An Optimized Sports Betting Approach

## Overview of the Project
This project uses the `NBA_api` to gather data on the outcomes of NBA games from the 2020-2024 seasons. The data was used to create a characteristic feature set that trained a supervised machine learning model to predict the outcome of a home team winning a given game in the 2024-2025 season. Using these predictions, a betting strategy was developed to maximize the expected value of profit, which was solved using a mixed-integer linear programming (MILP) formulation.


## Installation
To install the necessary packages to run this project, use the following command:
```bash
pip install nba_api numpy pandas matplotlib scikit-learn pulp json difflib requests time
```

## Usage
To create the datasets for the project, run the following command:
```bash
python win_loss_data_prep.py
```
This process may take several hours due to API call limits and the large volume of data being requested. However, the data is already saved in the `csv_data_file` folder, so this step can be skipped. 

The following files in the `csv_data_files` folder are used to train the machine learning model containing the feature set for the model: 
- `2020-25_scheduleFrame.csv`
- `2020-25_gameLogFrame.csv`
- `2020-25_nbaHomeWinLossModelDataSet.csv`

The following files were created to ensure functionality of the code but are not used in the final project:
- `2024-25_scheduleFrame.csv`
- `2024-25_gameLogFrame.csv`
- `2024-25_nbaHomeWinLossModelDataSet.csv`

To train the machine learning model and evaluate its performance, run the following command after specifying the model type in the main function:

```bash
python win_loss_data_model.py
```
In addition to the previously mentioned data files, the following file is required to generate the moneyline multipliers for the optimization strategy:
- `betting_odds.csv`

The output of the modeling script creates four plots that show the Receiver Operating Characteristic (ROC) curve and the time series of correct predictions for the model, with predicted probabilities of the home team winning on the y-axis. These plots are generated for both the training and validation datasets and are saved in the `model_ROC_and_accuracy_plots` folder.

Additionally, a csv file labled:
- `<model_type>_optimal_wagers.csv`

Details the optimal wagers as determined by the optimization strategy.


## Context for the Project
This project was completed as a final project for ASEN 6519: Optimization Algorithms and Applications at the University of Colorado Boulder. In addition to the code base, a final paper was written, and a presentation was given to the class detailing the data collection, feature extraction, model training, optimization strategy, and results. 

The final paper is included in the repository as `Bagley_Aidan_ASEN6519_Final_Project_Mixed_Integer_Linear_Parlays.pdf`, and the presentation is included as `Mixed-Integer Linear Parlays, An Optimized Sports Betting Approach.pdf`.

The project was completed by Aidan Bagley, a graduate student in the Aerospace Engineering Sciences department at the University of Colorado Boulder. The project was completed in the Fall 2024 semester. 

The project was driven by the desire to apply optimization techniques to a real-world problem, particularly in light of the recent legalization of sports betting in Colorado. It also provided an opportunity to explore the combination of supervised machine learning, API data collection, and optimization techniques to develop a profitable betting strategy.


## Solution Approach
### Data Collection
The [NBA API](https://github.com/swar/nba_api.git) provides an API client that allows for the extraction of game-by-game data from completed NBA games, spanning multiple seasons. For this project, data was collected from the 2020–2025 seasons using two key endpoints:

- `cumestatsteamgames`
- `cumestatsteam`

This data includes details about game outcomes and team statistics for each individual game. Once collected, the data was cleaned and processed using the `pandas` library in Python to create a feature set for the machine learning model.

### Feature Extraction

In this project, the following feature set was used:

- **Home Team Win Percentage**: The percentage of games the home team wins when playing at home.
- **Away Team Win Percentage**: The percentage of games the away team wins when playing away.
- **Total Win Percentage**: The overall win percentage of each team.
- **Offensive Efficiency**: A metric to evaluate a team's offensive performance.
- **Rolling Offensive Efficiency**: A time-based rolling metric of offensive performance.
- **Rolling Scoring Margin**: A time-based rolling metric of the average scoring margin.
- **Number of Rest Days**: The number of rest days the home team has had before the game.

These features were recommended in a tutorial provided within the [nba_api repository](https://github.com/swar/nba_api).

The rolling statistics are important to alleviate the stiffness of the dataset as the season progresses. They aim to encapsulate the recent performance of a team, providing a more accurate representation of the team's current state.


### Model Training
This project sought to compare two different supervised machine learning models to predict the outcome of a game. The first model was a logistic regression model and the second was a a Multi-Layer Perceptron (MLP) model. The models were trained on the same feature set using a 75/25 train/test split. The models were validated against results from the 2024-2025 season. Importantly, in addition to the accuracy of the models, the models were evaluated on their ability to predict probabilities of a home team win. This was important for the optimization strategy.


### Model Training

This project aimed to compare two different supervised machine learning models for predicting the outcome of a game:

1. **Logistic Regression**: A model used for binary classification tasks, predicting whether the home team will win or lose. Logistic regression estimates the probability of a binary outcome by fitting a linear relationship between the features and the log-likelihood of the target variable.
   
2. **Multi-Layer Perceptron (MLP)**: A type of neural network model used for binary classification tasks, predicting whether the home team will win or lose. MLPs are more flexible than logistic regression and can model complex, non-linear relationships between the features and the target variable through multiple layers of neurons.

Both models were trained using the same feature set, with a 75/25 train/test split. The models were validated against results from the 2024-2025 season.

In addition to evaluating the models based on accuracy, they were also assessed on their ability to predict the probabilities of a home team win. This was a key factor in optimizing the expected value of a wager, which is discussed in the next section.


### Optimization Strategy

The goal of the optimization strategy was to maximize the expected value of profit from betting on NBA games. The optimization problem was formulated as a mixed-integer linear programming (MILP) problem. 

The decision variables were whether to bet on a given game, and if so, how much money to bet. The objective function represented the expected value of profit from betting on the games and was calculated using the following formula:

\[
    \mathbb{E}[b_i] = \hat{p_i} \cdot (b_i \cdot \mathcal{R}_i) - (1-\hat{p_i}) \cdot b_i
\]

Where:
- \(\hat{p_i}\) is the predicted probability of the home team winning the game \(i\),
- \(b_i\) is the amount of money bet on game \(i\),
- \(\mathcal{R}_i\) is the moneyline multiplier for the home team winning game \(i\).

The constraints for the optimization problem were as follows:
- The total amount of money bet could not exceed a set budget.
- The total number of bets could not exceed a specified limit.
- The amount of money bet on a single game could not exceed a set maximum bet.
- The amount of money bet on a single game must be non-negative.
- The decision variable for betting on a game must be binary (i.e., either bet or not bet).



## Results

The results of the project were mixed (no pun intended). The logistic regression model outperformed the MLP model in terms of overall accuracy and its ability to differentiate between the positive and zero classes. This suggests that it was more effective in identifying home team wins and distinguishing between a home team win and loss. 

Additionally, the logistic regression model produced much less extreme probabilities, which helped in the subsequent expected value calculation. In contrast, the MLP had a much larger cross-entropy loss, indicating that it was less confident in its predictions. 

While the MLP had a higher recall value, implying it was able to identify home losses more effectively, this was likely due to the model simply classifying more games into the zero class, which is not necessarily indicative of a better model. 

Overall, the results comparing these models are tabulated in Tables 2, 3, 4, and 5, with the associated Figures 2 and 3 in the attached paper.


The optimization strategy was able to produce a betting strategy that, when given a $1000 budget over four separate days, resulted in a loss of $353.33. While this is not an ideal outcome, it is worth noting that the optimizer typically chose to wager very large amounts on games with moderate probabilities of winning but extremely large profit multipliers. 

In other words, the optimizer tended to "go big" when there was a chance to win a substantial amount of money. This behavior aligns with the objective function, which did not incentivize taking safer wagers with higher probabilities of winning but lower profit multipliers. This is an expected limitation of the formulation and is often referred to as gambling bias. 

Moderately probable events with the potential for high payouts are often overvalued by gamblers, and this bias is reflected in the optimizer's strategy.


## Contributions

The data collection, feature extraction, and logistic regression model training were heavily guided by a tutorial in the NBA_api documentation titled "Home Team Win-Loss Modeling." While the tutorial was closely followed, the code was adapted to fit the specific needs of this project. 

The MLP model training and optimization strategy were developed independently. The optimization strategy was implemented with the help of the PuLP library in Python. 

The final paper and presentation were completed independently.
