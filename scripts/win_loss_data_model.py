### This script is used to generate either a logistic regression or neural network model to predict the outcome of NBA games.
### The model is then used to generate optimal wagers based on the model's predictions and the odds of the games.
### This model outputs the results of the model evaluation, the ROC curve, and the optimal wagers for the given odds.


# Import pandas data frames
import pandas as pd

# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Import pulp for linear programming
import pulp as p

# Import scikit-learn modules
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import (accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    classification_report,
    roc_curve)

def print_results(y_test, y_pred, y_pred_proba):
    """Prints the results of the model evaluation."""
    ### Input:  y_test: The true labels
    ###         y_pred: The predicted labels
    ###         y_pred_proba: The predicted probabilities
    
    ### Output: cross_entropy: The cross-entropy loss
    ###        auc_score: The AUC-ROC score
    
    
    # Accuracy --> The proportion of correct predictions
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Precision --> The proportion of positive identifications that were actually correct
    precision = precision_score(y_test, y_pred, average="binary")
    print(f"Precision: {precision:.2f}")
    
    # Recall --> The proportion of actual positives that were correctly identified
    recall = recall_score(y_test, y_pred, average="binary")
    print(f"Recall: {recall:.2f}")
    
    # Specificity --> The proportion of actual negatives that were correctly identified
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  # Confusion matrix unpacking
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.2f}")
    
    # F1 Score --> The harmonic mean of precision and recall
    f1 = f1_score(y_test, y_pred, average="binary")
    print(f"F1 Score: {f1:.2f}")
    
    # Cross-Entropy Loss --> The measure of the model's confidence in its predictions
    cross_entropy = log_loss(y_test, y_pred_proba)
    print(f"Cross-Entropy Loss: {cross_entropy:.2f}")
    
    # AUC-ROC Score --> The area under the ROC curve, ability to differentiate between classes
    auc_score = roc_auc_score(y_test.values.ravel(), y_pred_proba[:,1])
    print(f"AUC-ROC: {auc_score:.2f}")
    
    # Full Classification Report --> Precision, Recall, F1, Support
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print('-'*50)
    
    return cross_entropy, auc_score
    
    
def plot_ROC(y_test, y_pred_proba, auc_score, cross_entropy, model_name, data_type):
    """Plots the ROC curve for the model."""
    ### Input:  y_test: The true labels
    ###         y_pred_proba: The predicted probabilities
    ###         auc_score: The AUC-ROC score
    ###         cross_entropy: The cross-entropy loss
    ###         model_name: The name of the model
    ###         data_type: The type of data (training or validation)
    
    ### Output: None
    
    # Plot Positive Class ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
         
    fig = plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')
    plt.plot(fpr, tpr, lw=2, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f}, CE Loss = {cross_entropy:.2f})')
    plt.scatter(fpr, tpr, s=10, color='red', alpha=0.7, label='Threshold Points')
    
    # Add labels
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    # Add title if training/validation for Logistic Regression/Neural Network
    data_type = "training" if data_type == 'training' else "validation"
    data_string = "Training" if data_type == 'training' else "Validation"
    models_str = "Logistic Regression" if model_name == 'logistic_regression' else "Neural Network"
    plt.title(f'{data_string} Data: {models_str} ROC Curve', fontsize=14)     
    
    # Add legend and grid
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate the threshold points     
    for i in range(0, len(thresholds), 30):  # Annotate every 10th point
        plt.text(fpr[i] + 0.02, tpr[i] - 0.02, f'{thresholds[i]:.2f}', fontsize=10, alpha=0.7)
     
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{model_name}_{data_type}_roc_curve.png')
        


def generate_logistic_model(data):
    """Generates a logistic regression model using the data."""
    ### Input:  data: The dataset to use for the model
    ###
    ### Output: model: The trained logistic regression model
    ###         training_probability_df: The training probability dataframe
    ###         validation_probability_df: The validation probability dataframe
    
    # Create validation data on 2024-25 season
    validation = data[data['SEASON'] == '2024-25']

    # Create model data set for all seasons except 2024-25
    modelData = data[data['SEASON'] != '2024-25'].sample(frac=1)

    # Input and Output Variables
    # Input Variables:  HOME_LAST_GAME_OE, HOME_LAST_GAME_HOME_WIN_PCTG, HOME_NUM_REST_DAYS, HOME_LAST_GAME_AWAY_WIN_PCTG, HOME_LAST_GAME_TOTAL_WIN_PCTG, HOME_LAST_GAME_ROLLING_SCORING_MARGIN, HOME_LAST_GAME_ROLLING_OE,
    #                   AWAY_LAST_GAME_OE, AWAY_LAST_GAME_HOME_WIN_PCTG, AWAY_NUM_REST_DAYS, AWAY_LAST_GAME_AWAY_WIN_PCTG, AWAY_LAST_GAME_TOTAL_WIN_PCTG, AWAY_LAST_GAME_ROLLING_SCORING_MARGIN, AWAY_LAST_GAME_ROLLING_OE
    #
    # Output Variable: HOME_W

    X = modelData.drop(['HOME_W','SEASON', 'GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'],axis=1) # input variables for model
    y = modelData['HOME_W'] # output is probability of home team winning

    # Turn X into float
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()

    # Turn y into float
    y = y.apply(pd.to_numeric, errors='coerce')
    y = y.dropna()
    y = y[X.index]

    # Randomly split 1/4 of data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25)

    # Create a StandardScaler() object called scaler
    scaler = preprocessing.StandardScaler()

    # Makes the training data have mean 0 and variance 1 to ensure features are on the same scale
    scaler.fit(X_train)

    # Transform the training data
    scaled_data_train = scaler.transform(X_train)

    # Transform the testing data
    scaled_data_test = scaler.transform(X_test)

    # Create a Logistic Regression model object
    model = LogisticRegression()

    # Fit the model to the scaled training data
    model.fit(scaled_data_train,y_train)
    
    # Test Set Review
    y_pred = model.predict(scaled_data_test)
    y_pred_proba = model.predict_proba(scaled_data_test)
    
    # Print Testing Model Results    
    cross_entropy, auc_score = print_results(y_test, y_pred, y_pred_proba)
        
    # Plot ROC Curve
    plot_ROC(y_test, y_pred_proba, cross_entropy,  auc_score, 'logistic_regression', 'training')

    # Locate the gameIDs for the probabilities to match with the game logs
    gameIDs = modelData.loc[X_test.index, 'GAME_ID']
    
    # Combine test input indices with probabilities
    training_probability_df = pd.DataFrame({'Index': gameIDs, 'Model_Predicted_Win': y_pred, 'Probability_Home_Team_Wins': y_pred_proba[:,1]}).set_index('Index')
    
    #############################################################################################################################################################
    # Validate against 2024-25 season using same input variables and standard scaling

    # Standard Scaling Prediction Variables
    validation_X = validation.drop(['HOME_W','SEASON','GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'],axis=1)
    validation_y = validation['HOME_W']

    # Turn validation_X into float
    validation_X = validation_X.apply(pd.to_numeric, errors='coerce')
    validation_X = validation_X.dropna()

    # Turn validation_y into float
    validation_y = validation_y.apply(pd.to_numeric, errors='coerce')
    validation_y = validation_y.dropna()
    validation_y = validation_y[validation_X.index]

    # Create a standard scaler object
    scaler = preprocessing.StandardScaler()

    # Fit the scaler to the validation input
    scaler.fit(validation_X)
    scaled_val_data = scaler.transform(validation_X)

    # How the model performs on validation data
    validation_y_pred = model.predict(scaled_val_data)
    validation_y_pred_proba = model.predict_proba(scaled_val_data)
    
    # Print Validation Model Results
    cross_entropy, auc_score = print_results(validation_y, validation_y_pred, validation_y_pred_proba)
    
    # Plot Positive Class ROC Curve
    plot_ROC(validation_y, validation_y_pred_proba, cross_entropy,  auc_score, 'logistic_regression', 'validation')
    
    # Locate the gameIDs for the probabilities
    gameIDs = validation.loc[validation_X.index, 'GAME_ID']
    
    # Combine validation input indices with probabilities
    validation_probability_df = pd.DataFrame({'Index': gameIDs, 'Model_Predicted_Win': validation_y_pred, 'Probability_Home_Team_Wins': validation_y_pred_proba[:,1]}).set_index('Index')
        
    # Return the model and the probability dataframes
    return model, training_probability_df, validation_probability_df
    
    
def generate_neural_network_model(data):
    """Generates a neural network model using the data."""
    ### Input:  data: The dataset to use for the model
    ###
    ### Output: model: The trained logistic regression model
    ###         training_probability_df: The training probability dataframe
    ###         validation_probability_df: The validation probability dataframe
    
    # Create validation data on 2024-25 season
    validation = data[data['SEASON'] == '2024-25']
    
    # Create model data set for all seasons except 2024-25
    modelData = data[data['SEASON'] != '2024-25'].sample(frac=1)
    
    # Input and Output Variables
    # Input Variables:  HOME_LAST_GAME_OE, HOME_LAST_GAME_HOME_WIN_PCTG, HOME_NUM_REST_DAYS, HOME_LAST_GAME_AWAY_WIN_PCTG, HOME_LAST_GAME_TOTAL_WIN_PCTG, HOME_LAST_GAME_ROLLING_SCORING_MARGIN, HOME_LAST_GAME_ROLLING_OE,
    #                   AWAY_LAST_GAME_OE, AWAY_LAST_GAME_HOME_WIN_PCTG, AWAY_NUM_REST_DAYS, AWAY_LAST_GAME_AWAY_WIN_PCTG, AWAY_LAST_GAME_TOTAL_WIN_PCTG, AWAY_LAST_GAME_ROLLING_SCORING_MARGIN, AWAY_LAST_GAME_ROLLING_OE
    #
    # Output Variable: HOME_W
    
    X = modelData.drop(['HOME_W','SEASON', 'GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'],axis=1) # input everything except the HOME_W and SEASON columns
    y = modelData['HOME_W'] # output is probability of home team winning
    
    # Turn X into float
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    
    # Turn y into float
    y = y.apply(pd.to_numeric, errors='coerce')
    y = y.dropna()
    y = y[X.index]

    # Randomly split 1/4 of data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25) # if reproducable desired, random_state=42)
    
    # Create a StandardScaler() object called scaler
    scaler = preprocessing.StandardScaler()
    
    # Makes the training data have mean 0 and variance 1 to ensure features are on the same scale
    scaler.fit(X_train)
    
    # Transform the training data
    scaled_data_train = scaler.transform(X_train)
    
    # Transform the testing data
    scaled_data_test = scaler.transform(X_test)
    
    # Create a Multi-Lyaer Perciptron Neural Network model object
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000000, alpha=0.0001,
                            solver='adam', verbose=False, tol = 0.000000001, random_state=42, learning_rate_init=.001)

    # Fit the model to the scaled training data
    model.fit(scaled_data_train, y_train)
    
    # Test set review
    y_pred = model.predict(scaled_data_test)
    y_pred_proba = model.predict_proba(scaled_data_test)
    
    # Print Testing Model Results
    cross_entropy, auc_score = print_results(y_test, y_pred, y_pred_proba)
    
    # Locate the gameIDs for the probabilities to match with the game logs
    plot_ROC(y_test, y_pred_proba, cross_entropy,  auc_score, 'neural_network', 'training')
    
    # Locate the gameIDs for the probabilities
    gameIDs = modelData.loc[X_test.index, 'GAME_ID']
    
    # Combine test input indices with probabilities
    training_probability_df = pd.DataFrame({'Index': gameIDs, 'Model_Predicted_Win': y_pred, 'Probability_Home_Team_Wins': y_pred_proba[:,1]}).set_index('Index')
    
    #############################################################################################################################################################
    # Validate against 2024-25 season using same input variables and standard scaling
    
    # Standard Scaling Prediction Variables
    validation_X = validation.drop(['HOME_W','SEASON','GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'],axis=1)
    validation_y = validation['HOME_W']
    
    # Turn validation_X into float
    validation_X = validation_X.apply(pd.to_numeric, errors='coerce')
    validation_X = validation_X.dropna()
    
    # Turn validation_y into float
    validation_y = validation_y.apply(pd.to_numeric, errors='coerce')
    validation_y = validation_y.dropna()
    validation_y = validation_y[validation_X.index]
    
    # Create a standard scaler object
    scaler = preprocessing.StandardScaler()
    
    # Fit the scaler to the validation input
    scaler.fit(validation_X)
    scaled_val_data = scaler.transform(validation_X)
    
    # How the model performs on unseen data
    validation_y_pred = model.predict(scaled_val_data)
    validation_y_pred_proba = model.predict_proba(scaled_val_data)
    
    # Print Validation Model Results
    cross_entropy, auc_score = print_results(validation_y, validation_y_pred, validation_y_pred_proba)
    
    # Plot Positive Class ROC Curve
    plot_ROC(validation_y, validation_y_pred_proba, cross_entropy,  auc_score, 'neural_network', 'validation')
    
    # Locate the gameIDs for the probabilities
    gameIDs = validation.loc[validation_X.index, 'GAME_ID']
    
    # Combine validation input indices with probabilities
    validation_probability_df = pd.DataFrame({'Index': gameIDs, 'Model_Predicted_Win': validation_y_pred, 'Probability_Home_Team_Wins': validation_y_pred_proba[:,1]}).set_index('Index')
    
    # Return the model and the probability dataframes
    return model, training_probability_df, validation_probability_df
    
    
def compare_model_to_actual_results(gameLogs, probability_dfs, model):
    """Compares the model's predictions to the actual results."""
    ### Input:  gameLogs: The game logs dataframe
    ###         probability_dfs: The list of probability dataframes
    ###         model: The model name
    ###
    ### Output: probability_df: The probability dataframe for the validation data
    
    def plot_results(probability_df, model, i):
        """Plots the results of the model's predictions."""
        ### Input:  probability_df: The probability dataframe
        ###         model: The model name
        ###         i: The index of the probability dataframe either training or validation
        ###
        ### Output: None
                   
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot the bar chart
        probability_df['Probability_Home_Team_Wins'].plot(
            kind='bar',
            color=probability_df['Correct_Model_Predictions'].map({True: 'green', False: 'red'}),
            width=1.0,
            align='center',
            alpha=0.7,
            ax=ax
        )

        # Add a horizontal line at 0.5
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.2, label='50% Threshold')

        # Set x-axis ticks and labels for better readability
        xticks = ax.get_xticks()
        max = 80 if i == 0 else 10
        ax.set_xticks(xticks[::max])  # Show every 80th tick for readability
        ax.set_xticklabels(
            probability_df['GAME_DATE'].dt.strftime('%Y-%m-%d').iloc[::max], 
            rotation=45, 
            ha='right', 
            fontsize=10
        )

        # Add gridlines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set axis labels and title
        plt.xlabel('Game Date', fontsize=12)
        plt.ylabel('Probability Home Team Wins', fontsize=12)

        # Dynamic title
        data_type = "training" if i == 0 else "validation"
        data_string = "Training" if i == 0 else "Validation"
        if model == 'logistic_regression':
            plt.title(f'{data_string} Data: Probability of Home Team Winning with Correct Predictions Highlighted using Logistic Regression', fontsize=14)
        else:
            plt.title(f'{data_string} Data: Probability of Home Team Winning with Correct Predictions Highlighted using Neural Network', fontsize=14)
        
        # Percent correct predictions
        percent_correct = sum(probability_df['Correct_Model_Predictions']) / len(probability_df) * 100
        
        # Add legend
        correct_patch = plt.Line2D([0], [0], color='green', lw=4, label=f'Correct Predictions ({percent_correct:.2f}%)')
        incorrect_patch = plt.Line2D([0], [0], color='red', lw=4, label=f'Incorrect Predictions ({100 - percent_correct:.2f}%)')
        ax.legend(handles=[correct_patch, incorrect_patch], loc='upper left', fontsize=10)

        # Add tight layout and save the figure
        plt.tight_layout()
        plt.savefig(f'model_ROC_and_accuracy_plots/{model}_{data_type}_probability_predictions.png')
        plt.close()
    
    
    # Convert both to a common type
    gameLogs['GAME_ID'] = gameLogs['GAME_ID'].astype(str)  # Convert to string
    gameLogs = gameLogs[gameLogs['HOME_FLAG'] == '1']  # Filter out games where the home team isn't at home
    
    # For training and validation probability dataframes
    i = 0
    for probability_df in probability_dfs:            
        probability_df.index = probability_df.index.astype(str)  # Ensure index is string
            
        # Merge the probability dataframe with the game logs on the GAME_ID
        probability_df = pd.merge(gameLogs, probability_df, left_on='GAME_ID', right_index=True, how='left')

        # Compare the probability of the home team winning to the actual result
        probability_df['Actual_Home_Win'] = probability_df['W_HOME'] == '1'
        probability_df['Correct_Model_Predictions'] = probability_df['Model_Predicted_Win'] == probability_df['Actual_Home_Win']

        # Sort the game logs by date
        probability_df['GAME_DATE'] = pd.to_datetime(probability_df['GAME_DATE'])
        probability_df = probability_df.sort_values(by='GAME_DATE')
        
        # Drop rows with missing values
        probability_df = probability_df.dropna(subset=['Probability_Home_Team_Wins'])
        probability_df = probability_df.dropna(subset=['Correct_Model_Predictions'])
        
        # Plot the results
        plot_results(probability_df, model, i)
        
        i += 1
                      
    return probability_df


def combine_odds_with_ML_probabilities(filename, gameLogs_probability_df):
    """Creates the optimal wagers for the given odds using model predictions."""
    ### Input:  filename: The filename of the odds 
    ###         gameLogs_probability_df: The game logs probability dataframe
    ###
    ### Output: betting_odds.csv: The optimal wagers for the given odds and model predictions
    
    # Create example dataframe of odds csv with columns
    #   Index(['DATE', 'AWAY', 'HOME', 'AWAY_MONEYLINE', 'HOME_MONEYLINE'], dtype='object')
    odds = pd.read_csv(filename, delimiter=',', header=0, dtype=str)
    
    # Convert Date to datetime
    odds['DATE'] = pd.to_datetime(odds['DATE'])
    
    # Convert columns to strings and strip whitespace
    string_columns = ['AWAY', 'HOME']
    for col in string_columns:
        odds[col] = odds[col].astype(str).str.strip()

    # Convert moneyline columns to numeric
    moneyline_columns = ['HOME_MONEYLINE', 'AWAY_MONEYLINE']
    for col in moneyline_columns:
        odds[col] = pd.to_numeric(odds[col], errors='coerce')

    # Ensure the game logs probability dataframe has the correct data types
    gameLogs_probability_df['GAME_DATE'] = pd.to_datetime(gameLogs_probability_df['GAME_DATE'])
    gameLogs_probability_df['NICKNAME'] = gameLogs_probability_df['NICKNAME'].str.strip()
    
    # Merge the odds with the probability dataframe
    game_ids = []
    for _, row in odds.iterrows():
        # Find if there is a match between probability dataframe and odds dataframe
        matches = gameLogs_probability_df[
            (gameLogs_probability_df['GAME_DATE'] == row['DATE']) &
            (gameLogs_probability_df['NICKNAME'] == row['HOME']) &
            (gameLogs_probability_df['HOME_FLAG'] == '1')
        ]
        # If there is a match, append the game ID
        if len(matches) == 1:
            game_ids.append(matches['GAME_ID'].values[0])
        else:
            game_ids.append(None)
            
    # Add the game IDs to the odds dataframe
    odds['GAME_ID'] = game_ids
        
    # Merge the odds with the game logs do create the optimal wagers
    odds = pd.merge(odds, gameLogs_probability_df, on='GAME_ID', how='left')
    
    # Reset index for odds
    odds = odds.reset_index(drop=True)
    
    # Lets pick at least 5 games to bet on for each day of games
    days = odds['DATE'].unique()
    gameIdx = 0
    
    # For each day odds are available
    for day in days:
        
        # Create a new data frame for just the games on that day
        day_odds = odds[odds['DATE'] == day]
        day_odds.reset_index(drop=True, inplace=True)

        # Solve the linear program for the optimal wagers on that day
        z = solve_LP(day_odds)
        
        # Evaluate the optimal wagers against real results 
        for i in range(len(day_odds)):
            # Grab the optimizer results for the game
            zopt = z[i]
            xopt = 1 if zopt > 0.1 else 0
            
            # Update the odds dataframe with the optimal wagers
            odds.loc[gameIdx, 'Wager_Yes_No'] = xopt
            odds.loc[gameIdx, 'Optimal_Wager'] = zopt
            
            # Calculate the potential return and actual return for the optimal wagers if positive class is chosen
            if odds.loc[i, 'Probability_Home_Team_Wins'] > 0.5:
                odds.loc[gameIdx, 'Potential_Return'] = zopt * potential_win(odds.loc[gameIdx, 'HOME_MONEYLINE'])
                odds.loc[gameIdx, 'Winning_Wager'] = odds.loc[gameIdx, 'Actual_Home_Win']

            # Calculate the potential return and actual return for the optimal wagers if negative class is chosen
            else:
                odds.loc[gameIdx, 'Potential_Return'] = zopt * potential_win(odds.loc[gameIdx, 'AWAY_MONEYLINE'])
                odds.loc[gameIdx, 'Winning_Wager'] = not odds.loc[gameIdx, 'Actual_Home_Win']
             
            # Did the model predict the correct outcome? Did the optimal wager win?
            odds.loc[gameIdx, 'Actual_Return'] = odds.loc[gameIdx, 'Winning_Wager'] * odds.loc[gameIdx, 'Potential_Return']

            # Increment the game index
            gameIdx += 1
            
    # Save the optimal wagers to a csv file
    odds = odds[['DATE', 'AWAY', 'HOME', 'AWAY_MONEYLINE', 'HOME_MONEYLINE', 'GAME_ID', 'CITY', 'NICKNAME',  'Model_Predicted_Win', 'Probability_Home_Team_Wins', 'Actual_Home_Win', 'Correct_Model_Predictions', 'Wager_Yes_No', 'Optimal_Wager', 'Potential_Return', 'Actual_Return']]
    odds.to_csv('optimal_wagers.csv', index=False)

    
def solve_LP(odds):
    """Solves the linear program for the optimal wagers."""
    ### Input:  odds: The odds dataframe
    ###
    ### Output: z: The optimal wagers given the avaiable odds
    
    # Create a linear program to optimize 3-leg parlays
    prob = p.LpProblem("Parlay", p.LpMaximize)
    max_b = 1000  # Maximum wager per game
    num_wagers = len(odds)  # Number of wagers

    # Variables
    x = p.LpVariable.dicts("x", range(len(odds)), cat=p.LpBinary)  # Binary decision variables
    b = [p.LpVariable(f"b_{i}", lowBound=max_b/(4*num_wagers), upBound=max_b) for i in range(len(odds))]  # Continuous wagers
    z = [p.LpVariable(f"z_{i}", lowBound=0) for i in range(len(odds))]  # Slack variable to create linear program

    # Budget constraint
    prob += p.lpSum(z) == max_b, "Budget_Constraint"

    # Linearize z[i] = x[i] * b[i]
    for i in range(num_wagers):
        prob += z[i] <= b[i], f"UpperBound_b_{i}"
        prob += z[i] <= x[i] * max_b, f"UpperBound_xb_{i}"
        prob += z[i] >= b[i] - (1 - x[i]) * max_b, f"LowerBound_b_{i}"

    # Must wager on at least half of the games
    prob += p.lpSum(x) == int(num_wagers/2)  , "Three_Legs_Constraint"    

    # Objective function: Maximize expected value = prob_win * potential_win - prob_loss * wager
    expected_values = []
    
    # For each potential wager
    for i in range(len(odds)):
        # Probability of home team winning and home team losing
        prob_win = odds.at[i, 'Probability_Home_Team_Wins']
        prob_loss = 1 - prob_win
        
        # Moneyline odds for home and away teams
        home_moneyline = odds.at[i, 'HOME_MONEYLINE']
        away_moneyline = odds.at[i, 'AWAY_MONEYLINE']
        
        # Compute the expected value of the wager = (prob_winning_bet * winning_bet_multiplier - prob_losing_bet * wager) - wager
        if prob_win > 0.5:
            expected_value = prob_win * z[i]*potential_win(home_moneyline) - prob_loss*z[i] - z[i]
        else:
            expected_value = (1-prob_win) * z[i]*potential_win(away_moneyline) - prob_win*z[i] - z[i]
            
        expected_values.append(expected_value)

    # Objective function
    prob += p.lpSum(expected_values), "Objective"
    
    # Solve
    prob.solve(p.PULP_CBC_CMD(msg=False))
        
    # Return z numpy array
    z = np.array([p.value(z[i]) for i in range(len(odds))])
    
    # Return the optimal wagers
    return z
            

def potential_win(moneyLine):
    """Calculates the potential win for a given moneyline."""
    ### Input:  moneyLine: The moneyline odds
    ###
    ### Output: The potential winning multiplier for a given wager
    if moneyLine > 0:
        return (moneyLine / 100)
    else:
        return (100 / abs(moneyLine))            
         




if __name__ == '__main__':
    """Main function to load in feature sets, create SVML model, and solve MILP for optimal wagers"""
    # Load feature and gamelog sets
    feature_set_filename = 'csv_data_files/2020-25_nbaHomeWinLossModelDataset.csv'
    gameLogs_filename = 'csv_data_files/2020-25_gameLogs.csv'
    
    # Generate model
    modelname = 'logistic_regression'
    #modelname = 'neural_network'
    
    # Load feature set    
    data = pd.read_csv(feature_set_filename, delimiter=',', header=0, dtype=str)
        
    # Load original game logs
    gameLogs = pd.read_csv(gameLogs_filename, delimiter=',', header=0, dtype=str)
    
    # Generate model logistic regression
    if modelname == 'logistic_regression':    
        model, training_probability_df, validation_probability_df = generate_logistic_model(data)
        validation_gameLogs_probability_df = compare_model_to_actual_results(gameLogs, [training_probability_df, validation_probability_df], modelname)
        combine_odds_with_ML_probabilities('csv_data_files/betting_odds.csv', validation_gameLogs_probability_df)

    # Generate model neural network
    elif modelname == 'neural_network':
        model, training_probability_df, validation_probability_df = generate_neural_network_model(data)
        validation_gameLogs_probability_df = compare_model_to_actual_results(gameLogs, [training_probability_df, validation_probability_df], modelname)
        combine_odds_with_ML_probabilities('csv_data_files/betting_odds.csv', validation_gameLogs_probability_df)

    else:
        print('Invalid model name. Please choose either logistic_regression or neural_network.')
            