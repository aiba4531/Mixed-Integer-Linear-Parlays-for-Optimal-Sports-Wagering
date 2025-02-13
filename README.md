# Mixed-Integer Linear Parlays, An Optimized Sports Betting Approach

## Overview of the Project
This project uses the `NBA_api` to gather data on the outcomes of NBA games from the 2020-2024 seasons. The data is used to create a characteristic feature set that trains a supervised machine learning model to predict the outcome of a home team winning a given game in the 2024-2025 season. Using these predictions, a betting strategy is developed to maximize the expected value of profit, which was is solved using a mixed-integer linear programming (MILP) formulation.


## Installation
To install the necessary packages to run this project, use the following command:
```bash
pip install nba_api numpy pandas matplotlib scikit-learn pulp requests
```

Afterward, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/aiba4531/Mixed-Integer-Linear-Parlays-for-Optimal-Sports-Wagering.git
```

Naviate to the `Mixed-Integer-Linear-Parlays-for-Optimal-Sports-Wagering` directory and clone the NBA API repository to your local machine using the following command: 
```bash
git clone https://github.com/swar/nba_api.git
```
## Usage

### Running the Data Preparation Script
To create the datasets for the project, run the `win_loss_data_prep.py` script using the following command replacing `/path/to/python/interpreter` with the path to your python interpreter and `/path/to/Mixed-Integer-Linear-Parlays-for-Optimal-Sports-Wagering` with the path to the cloned repository.
```bash
/path/to/python/interpreter /path/to/Mixed-Integer-Linear-Parlays-for-Optimal-Sports-Wagering/scripts/win_loss_data_prep.py
```
This process may take several hours due to API call limits and the large volume of data being requested. However, the data is already saved in the `csv_data_file` folder, so this step can be skipped unless you want to update the data set for new results. 

The following files in the `csv_data_files` folder are used to train the machine learning model containing the feature set for the model: 
- `2020-25_scheduleFrame.csv`
- `2020-25_gameLogFrame.csv`
- `2020-25_nbaHomeWinLossModelDataSet.csv`

The following files were created to ensure functionality of the code but are not used in the final project:
- `2024-25_scheduleFrame.csv`
- `2024-25_gameLogFrame.csv`
- `2024-25_nbaHomeWinLossModelDataSet.csv`

### Running the Model Training and Optimization Script
To train the machine learning model, evaluate its performance, and generate the optimal wager set, run the following command after specifying the model type in the main function and replacing `/path/to/python/interpreter` with the path to your python interpreter and `/path/to/Mixed-Integer-Linear-Parlays-for-Optimal-Sports-Wagering` with the path to the cloned repository.

```bash
/path/to/python/interpreter /path/to/Mixed-Integer-Linear-Parlays-for-Optimal-Sports-Wagering/scripts/win_loss_data_model.py
```
In addition to the previously mentioned data files, the following file is required to generate the moneyline multipliers for the optimization strategy:
- `betting_odds.csv`

This data file was manually created using the **DraftKings** app to acquire the the moneyline odds on four separate days.

The output of the modeling script creates four plots that show the Receiver Operating Characteristic (ROC) curve and the time series of correct predictions for the model, with predicted probabilities of the home team winning on the y-axis. These plots are generated for both the training and validation datasets and are saved in the `model_ROC_and_accuracy_plots` folder.

Additionally, a csv file labled:
- `<model_type>_optimal_wagers.csv`

Details the optimal games to wager upon, the amount of money to wager, and the expected value of profit for each wager, and the resulting profit or loss from the strategy.


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

Notably, the data collection process was time-consuming due to the API's rate limits, which required a delay between requests to avoid being blocked. This resulted in a lengthy data collection process, which was mitigated by saving the data to CSV files for future use. Additionally, there are gaps in the data due to failed API requests, however, the data was still sufficient for the purposes of this project.

### Feature Extraction

In this project, the following feature set was used:

- **Home Team Win Percentage**: The percentage of games the home team wins when playing at home.
- **Away Team Win Percentage**: The percentage of games the away team wins when playing away.
- **Total Win Percentage**: The overall win percentage of each team.
- **Offensive Efficiency**: A metric to evaluate a team's offensive performance.
- **Rolling Offensive Efficiency**: A time-based rolling metric of offensive performance.
- **Rolling Scoring Margin**: A time-based rolling metric of the average scoring margin.
- **Number of Rest Days**: The number of rest days the home team has had before the game.

These features were recommended in a tutorial provided within the [NBA API](https://github.com/swar/nba_api.git).

The first three features are characteristic of a team's success rate both at home, away, and overall. The offensive efficiency and rolling offensive efficiency features provide insight into a team's offensive capabilities. Specifically, the rolling statistics are important to alleviate the stiffness of the dataset as the season progresses. They aim to encapsulate the recent performance of a team, providing a more accurate representation of the team's current state. Lastly, the number of rest days is included as a feature to account for the potential impact of fatigue on a team's performance.


### Model Training

This project aimed to compare two different supervised machine learning models for predicting the outcome of a game:

1. **Logistic Regression**: A model used for binary classification tasks, predicting whether the home team will win or lose. Logistic regression estimates the probability of a binary outcome by fitting a linear relationship between the features and the log-likelihood of the target variable.
   
2. **Multi-Layer Perceptron (MLP)**: A type of neural network model used for binary classification tasks, predicting whether the home team will win or lose. MLPs are more flexible than logistic regression and can model complex, non-linear relationships between the features and the target variable through multiple layers of neurons.

Both models were trained using the same feature set, with a 75/25 train/test split. The models were validated against results from the 2024-2025 season.

In addition to evaluating the models based on accuracy, they were also assessed on their ability to predict the probabilities of a home team win. This was a key factor in predicting the expected value of a wager, which is discussed in the next section.


### Optimization Strategy

The goal of the optimization strategy was to maximize the expected value of profit from betting on NBA games. The optimization problem was formulated as a mixed-integer linear programming (MILP) problem. 

The decision variables were whether to bet on a given game, and if so, how much money to bet. The objective function represented the expected value of profit from betting on the games and was calculated using the following formula:

$$
\mathbb{E}[b_i] = \hat{p_i} \cdot (b_i \cdot \mathcal{R}_i) - (1-\hat{p_i}) \cdot b_i
$$

Where:
- $\hat{p_i}$ is the predicted probability of the home team winning the game $i$,
- $b_i$ is the amount of money bet on game $i$,
- $\mathcal{R}_i$ is the moneyline multiplier for the home team winning game $i$.

The constraints for the optimization problem were as follows:
- The total amount of money bet could not exceed a set budget.
- The total number of bets must be equal to half of the available wagers.
- The amount of money bet on a single game could not exceed a set maximum bet.
- The amount of money bet on a single game must be at least the maximum buget divided by four times the number of wagers.
- The decision variable for betting on a game must be binary (i.e., either bet or not bet).

The optimization problem was solved using the `PuLP` library in Python, which provides an interface to various optimization solvers. The solution to the optimization problem provided the optimal betting strategy for maximizing the expected value of profit. 

Notably, the second and fourth constraints were an initial attempt at mitigating risk by forcing the optimizer to spread out smaller bets across multiple games. This was an attempt to reduce the impact of a single large bet on the overall outcome of the strategy. This was not entirely successful as discussed in the results section and slightly different than formulation presented in the paper. 


## Results

The results of the project were mixed (no pun intended). The logistic regression model outperformed the MLP model in terms of overall accuracy and its ability to differentiate between the positive and zero classes. This suggests that it was more effective in identifying home team wins and distinguishing between a home team win and loss. 

Additionally, the logistic regression model produced much less extreme probabilities, which helped in the subsequent expected value calculation. In contrast, the MLP predicted more extreme probabilities and had a much larger cross-entropy loss, indicating that it was less confident in its predictions. 

While the MLP had a higher recall value, implying it was able to identify home losses more effectively, this was likely due to the model simply classifying more games into the zero class, which is not necessarily indicative of a better model. 

Overall, the results comparing these models are tabulated in Tables 2, 3, 4, and 5, with the associated Figures 2 and 3 in the attached paper. Notably, the logistic regression model out performed the MLP and thus was used in the optimization strategy.


The optimization strategy was able to produce a betting strategy that, when given a $1000 budget over four separate days, resulted in a loss of $353.33 overall. While this is not an ideal outcome, it is worth noting that the optimizer typically chose to wager very large amounts on games with moderate probabilities of winning but extremely large profit multipliers. 

In other words, the optimizer tended to "go big" when there was a chance to win a substantial amount of money. This behavior aligns with the objective function, which did not incentivize taking safer wagers with higher probabilities of winning but lower profit multipliers. This is an expected limitation of the formulation and is often referred to as gambling bias. 

Moderately probable events with the potential for high payouts are often overvalued by gamblers, and this bias is reflected in the optimizer's strategy.


## Future Work
While the logistic regression model performed well in this project, there are several areas for future work and improvement:
- **Feature Engineering**: Additional features could be explored to improve the model's predictive power. For example, incorporating player statistics, team injuries, or other external factors could enhance the model's performance.
- **Model Selection**: Further exploration of ensamble machine learning models, such as random forests or gradient boosting, could be beneficial. These models may capture more complex relationships in the data and improve predictive accuracy. Furthermore, when using these models, less restrictive feature sets could be explored to see if the model can identify the most important features without directly specifying them.
- **Optimization Strategy**: The optimization strategy could be refined to include additional constraints or objectives. For example, incorporating a risk management component to limit the potential loss from a single bet could be beneficial. Additionally, exploring different objective functions that balance profit potential and risk could lead to a more robust betting strategy.
- **Feature Set Sample Size**: The model could benefit from a larger sample size of games to train on. Additional seasons of data could provide more insights into team performance and improve the model's predictive power.
- **Avaiable Wager Sample Size**: The optimization strategy could benefit from a larger sample size of available wagers. This would allow for more flexibility in the betting strategy and potentially improve the overall expected value of profit.

## Contributions

The data collection, feature extraction, and logistic regression model training were heavily guided by a tutorial in the NBA_api documentation titled "Home Team Win-Loss Modeling." While the tutorial was closely followed for creating and manipulating the data frames, the code was adapted to meet the specific needs of this project. 

The moneyline multipliers were manually collected from the **DraftKings** app and were not part of the tutorial.

The **MLP model training** and **optimization strategy** were developed independently by me. The optimization strategy was implemented with the help of the **PuLP library** in Python.

The final paper and presentation were completed independently, without collaboration, although I did refer to external resources like the `nba_api` tutorial for guidance.
