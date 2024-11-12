from nba_api.stats.static import teams
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam
import pandas as pd
import numpy as np
import json
import difflib
import time
import requests
import os


# Retry wrapper for API calls
def retry(func, retries = 3):
    def retry_wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                attempts += 1
                time.sleep(30)
                
    return retry_wrapper

# Get Season Schedule Function 
def getSeasonScheduleFrame(seasons,seasonType): 

    # Get date from string
    def getGameDate(matchup):
        return matchup.partition(' at')[0][:10]

    # Get Home team from string
    def getHomeTeam(matchup):
        return matchup.partition(' at')[2]

    # Get Away team from string
    def getAwayTeam(matchup):
        return matchup.partition(' at')[0][10:]

    # Match nickname from schedule to team table to find ID
    def getTeamIDFromNickname(nickname):
        return teamLookup.loc[teamLookup['nickname'] == difflib.get_close_matches(nickname,teamLookup['nickname'],1)[0]].values[0][0] 
    
    @retry
    def getRegularSeasonSchedule(season,teamID,seasonType):
        season = str(season) + "-" + str(season+1)[-2:] # Convert year to season format ie. 2020 -> 2020-21
        teamGames = cumestatsteamgames.CumeStatsTeamGames(league_id = '00',season = season ,
                                                                      season_type_all_star=seasonType,
                                                                      team_id = teamID).get_normalized_json()

        teamGames = pd.DataFrame(json.loads(teamGames)['CumeStatsTeamGames'])
        teamGames['SEASON'] = season
        return teamGames    
    
    # Get team lookup table
    teamLookup = pd.DataFrame(teams.get_teams())
    
    # Get teams schedule for each team for each season
    scheduleFrames = []  # Initialize an empty list to store DataFrames
    
    for season in seasons:
        for id in teamLookup['id']:
            scheduleFrames.append(getRegularSeasonSchedule(season, id, seasonType))
    
    # Concatenate all DataFrames in the list into a single DataFrame
    scheduleFrame = pd.concat(scheduleFrames, ignore_index=True)
    
            
    scheduleFrame['GAME_DATE'] = pd.to_datetime(scheduleFrame['MATCHUP'].map(getGameDate))
    scheduleFrame['HOME_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getHomeTeam)
    scheduleFrame['HOME_TEAM_ID'] = scheduleFrame['HOME_TEAM_NICKNAME'].map(getTeamIDFromNickname)
    scheduleFrame['AWAY_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getAwayTeam)
    scheduleFrame['AWAY_TEAM_ID'] = scheduleFrame['AWAY_TEAM_NICKNAME'].map(getTeamIDFromNickname)
    scheduleFrame = scheduleFrame.drop_duplicates() # There's a row for both teams, only need 1
    scheduleFrame = scheduleFrame.reset_index(drop=True)
            
    return scheduleFrame

# Get Single Game Metrics Function
def getSingleGameMetrics(gameID,homeTeamID,awayTeamID,awayTeamNickname,seasonYear,gameDate):
    
    @retry
    def getGameStats(teamID,gameID,seasonYear):
        gameStats = cumestatsteam.CumeStatsTeam(game_ids=gameID,league_id ="00",
                                               season=seasonYear,season_type_all_star="Regular Season",
                                               team_id = teamID).get_normalized_json()

        gameStats = pd.DataFrame(json.loads(gameStats)['TotalTeamStats'])

        return gameStats
    
    data = getGameStats(homeTeamID,gameID,seasonYear)
    data.at[1,'NICKNAME'] = awayTeamNickname
    data.at[1,'TEAM_ID'] = awayTeamID
    data.at[1,'OFFENSIVE_EFFICIENCY'] = (data.at[1,'FG'] + data.at[1,'AST'])/(data.at[1,'FGA'] - data.at[1,'OFF_REB'] + data.at[1,'AST'] + data.at[1,'TOTAL_TURNOVERS'])
    data.at[1,'SCORING_MARGIN'] = data.at[1,'PTS'] - data.at[0,'PTS']

    data.at[0,'OFFENSIVE_EFFICIENCY'] = (data.at[0,'FG'] + data.at[0,'AST'])/(data.at[0,'FGA'] - data.at[0,'OFF_REB'] + data.at[0,'AST'] + data.at[0,'TOTAL_TURNOVERS'])
    data.at[0,'SCORING_MARGIN'] = data.at[0,'PTS'] - data.at[1,'PTS']

    data['SEASON'] = seasonYear
    data['GAME_DATE'] = gameDate
    data['GAME_ID'] = gameID

    return data

# Get Game Logs Function
def getGameLogs(gameLogs, scheduleFrame):
     # Functions to prepare additional columns after gameLogs table loads
    
    # Sets flag to 1 if game was played at home win or loss
    def getHomeAwayFlag(gameDF):
        gameDF['HOME_FLAG'] = np.where((gameDF['W_HOME']==1) | (gameDF['L_HOME']==1),1,0)
        gameDF['AWAY_FLAG'] = np.where((gameDF['W_ROAD']==1) | (gameDF['L_ROAD']==1),1,0)
    
    # Gets the total number of games and counts the number of wins for each team
    def getTotalWinPctg(gameDF):
        gameDF['TOTAL_GAMES_PLAYED'] = gameDF.groupby(['TEAM_ID','SEASON'])['GAME_DATE'].rank(ascending=True)
        gameDF['TOTAL_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W'].cumsum()
        gameDF['TOTAL_WIN_PCTG'] = gameDF['TOTAL_WINS']/gameDF['TOTAL_GAMES_PLAYED']
        return gameDF.drop(['TOTAL_GAMES_PLAYED','TOTAL_WINS'],axis=1)

    # Gets the home win percentage for each team
    def getHomeWinPctg(gameDF):
        gameDF['HOME_GAMES_PLAYED'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['HOME_FLAG'].cumsum()
        gameDF['HOME_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W_HOME'].cumsum()
        gameDF['HOME_WIN_PCTG'] = gameDF['HOME_WINS']/gameDF['HOME_GAMES_PLAYED']
        return gameDF.drop(['HOME_GAMES_PLAYED','HOME_WINS'],axis=1)

    # Gets the away win percentage for each team
    def getAwayWinPctg(gameDF):
        gameDF['AWAY_GAMES_PLAYED'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['AWAY_FLAG'].cumsum()
        gameDF['AWAY_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W_ROAD'].cumsum()
        gameDF['AWAY_WIN_PCTG'] = gameDF['AWAY_WINS']/gameDF['AWAY_GAMES_PLAYED']
        return gameDF.drop(['AWAY_GAMES_PLAYED','AWAY_WINS'],axis=1)

    # Gets the rolling average offensive efficiency for each team taking average of last 3 games
    def getRollingOE(gameDF):
        gameDF['ROLLING_OE'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['OFFENSIVE_EFFICIENCY'].transform(lambda x: x.rolling(3, 1).mean())

    # Gets the rolling average scoring margin for each team taking average of last 3 games
    def getRollingScoringMargin(gameDF):
        gameDF['ROLLING_SCORING_MARGIN'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['SCORING_MARGIN'].transform(lambda x: x.rolling(3, 1).mean())

    # Gets the current game data, shifts it back one, then caluclates the time inbetween in days floating point
    def getRestDays(gameDF):
        gameDF['LAST_GAME_DATE'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['GAME_DATE'].shift(1)
        gameDF['NUM_REST_DAYS'] = (gameDF['GAME_DATE'] - gameDF['LAST_GAME_DATE'])/np.timedelta64(1,'D') 
        return gameDF.drop('LAST_GAME_DATE',axis=1)
    
    start = time.perf_counter_ns() # Track cell's runtime
    
    i = int(len(gameLogs)/2)
    
    # Initialize a list to store the DataFrames
    gameLogs_list = [gameLogs]

    while i < len(scheduleFrame):
        time.sleep(1)  # Sleep for 1 second to avoid API rate limit
        
        new_game_log = getSingleGameMetrics(
            scheduleFrame.at[i, 'GAME_ID'],
            scheduleFrame.at[i, 'HOME_TEAM_ID'],
            scheduleFrame.at[i, 'AWAY_TEAM_ID'],
            scheduleFrame.at[i, 'AWAY_TEAM_NICKNAME'],
            scheduleFrame.at[i, 'SEASON'],
            scheduleFrame.at[i, 'GAME_DATE']
        )
        
        # Append the new DataFrame to the list
        gameLogs_list.append(new_game_log)
        
        i += 1  # Increment the index
        
        end = time.perf_counter_ns()
        
        #Output time it took to load x amount of records
        if i%100 == 0:
            mins = ((end-start)/1e9)/60
            print(i,str(mins) + ' minutes')

        print("Iteration: ",i)

    # Concatenate all DataFrames in the list into a single DataFrame
    gameLogs = pd.concat(gameLogs_list, ignore_index=True)

    
    # Get Table Level Aggregation Columns
    getHomeAwayFlag(gameLogs)
    gameLogs = getHomeWinPctg(gameLogs)
    gameLogs = getAwayWinPctg(gameLogs)
    gameLogs = getTotalWinPctg(gameLogs)
    getRollingScoringMargin(gameLogs)
    getRollingOE(gameLogs)
    gameLogs = getRestDays(gameLogs)

    return gameLogs.reset_index(drop=True)

# Get Game Log Feature Set Function
def getGameLogFeatureSet(gameDF):

    def shiftGameLogRecords(gameDF):
        gameDF['LAST_GAME_OE'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['OFFENSIVE_EFFICIENCY'].shift(1)
        gameDF['LAST_GAME_HOME_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['HOME_WIN_PCTG'].shift(1)
        gameDF['LAST_GAME_AWAY_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['AWAY_WIN_PCTG'].shift(1)
        gameDF['LAST_GAME_TOTAL_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['TOTAL_WIN_PCTG'].shift(1)
        gameDF['LAST_GAME_ROLLING_SCORING_MARGIN'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['ROLLING_SCORING_MARGIN'].shift(1)
        gameDF['LAST_GAME_ROLLING_OE'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['ROLLING_OE'].shift(1)
    
    
    def getHomeTeamFrame(gameDF):
        homeTeamFrame = gameDF[gameDF['CITY'] != 'OPPONENTS']
        homeTeamFrame = homeTeamFrame[['LAST_GAME_OE','LAST_GAME_HOME_WIN_PCTG','NUM_REST_DAYS','LAST_GAME_AWAY_WIN_PCTG','LAST_GAME_TOTAL_WIN_PCTG','LAST_GAME_ROLLING_SCORING_MARGIN','LAST_GAME_ROLLING_OE','W','TEAM_ID','GAME_ID','SEASON']]

        colRenameDict = {}
        for col in homeTeamFrame.columns:
            if (col != 'GAME_ID') & (col != 'SEASON') :
                colRenameDict[col] = 'HOME_' + col 

        homeTeamFrame.rename(columns=colRenameDict,inplace=True)

        return homeTeamFrame

    def getAwayTeamFrame(gameDF):
        # Strip whitespace from CITY column
        gameDF['CITY'] = gameDF['CITY'].str.strip()    
        
        # Get the away team frame
        awayTeamFrame = gameDF[gameDF['CITY'] == 'OPPONENTS']
        awayTeamFrame = awayTeamFrame[['LAST_GAME_OE','LAST_GAME_HOME_WIN_PCTG','NUM_REST_DAYS','LAST_GAME_AWAY_WIN_PCTG','LAST_GAME_TOTAL_WIN_PCTG','LAST_GAME_ROLLING_SCORING_MARGIN','LAST_GAME_ROLLING_OE','TEAM_ID','GAME_ID','SEASON']]

        colRenameDict = {}
        for col in awayTeamFrame.columns:
            if (col != 'GAME_ID') & (col != 'SEASON'):
                colRenameDict[col] = 'AWAY_' + col 

        awayTeamFrame.rename(columns=colRenameDict,inplace=True)

        return awayTeamFrame
    
    shiftGameLogRecords(gameLogs)
    awayTeamFrame = getAwayTeamFrame(gameLogs)
    homeTeamFrame = getHomeTeamFrame(gameLogs)
    
    return pd.merge(homeTeamFrame, awayTeamFrame, how="inner", on=[ "GAME_ID","SEASON"]).drop(['GAME_ID','AWAY_TEAM_ID','HOME_TEAM_ID'],axis=1)


# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df



# Get ScheduleFrame
#seasons = [2020,2021,2022]
seasons = [2020]
seasonType = 'Regular Season'


# Check if the data is already saved
if os.path.exists('scheduleFrame.csv'):
    # Load the saved CSV file
    scheduleFrame = pd.read_csv('scheduleFrame.csv')
    scheduleFrame = clean_column_names(scheduleFrame)
    print("Loaded scheduleFrame from CSV.")
else:    
    # Create a new scheduleFrame
    print("Loading scheduleFrame...")
    start = time.perf_counter_ns() # Track cell's runtime
    scheduleFrame = getSeasonScheduleFrame(seasons,seasonType)
    end = time.perf_counter_ns()

    secs = (end-start)/1e9
    mins = secs/60
    print("Time to load scheduleFrame: ",str(mins) + ' minutes')
    
    # Save the scheduleFrame to CSV
    scheduleFrame.to_csv('scheduleFrame.csv', index=False)
    print("Saved scheduleFrame to CSV.")

# Display the first 5 rows of the scheduleFrame
print('-'*50)
print("ScheduleFrame: ")
print(scheduleFrame.head(5))
print('-'*50)


# Now continue with gameLogs
if os.path.exists('gameLogs.csv'):
    # Load the saved CSV file
    gameLogs = pd.read_csv('gameLogs.csv')
    gameLogs = clean_column_names(gameLogs)
    print("Loaded gameLogs from CSV.")
else:
    # Create a new gameLogs
    gameLogs = pd.DataFrame()
    gameLogs = getGameLogs(gameLogs, scheduleFrame)

    # Save the gameLogs to CSV
    gameLogs.to_csv('gameLogs.csv', index=False)
    print("Saved gameLogs to CSV.")

print('-'*50)
print("GameLogs: ")
print(gameLogs.head(5))
print('-'*50)


if os.path.exists('gameLogFeatureSet.csv'):
    gameLogFeatureSet = pd.read_csv('gameLogFeatureSet.csv')
    print("Loaded gameLogFeatureSet from CSV.")
else:
    gameLogFeatureSet = getGameLogFeatureSet(gameLogs)
    gameLogFeatureSet.to_csv('gameLogFeatureSet.csv', index=False)
    print("Saved gameLogFeatureSet to CSV.")

print('-'*50)
print("GameLogFeatureSet: ")
print(gameLogFeatureSet.head(5))
print('-'*50)

