### This script contains functions to load and prepare the data for the NBA Home Win Loss Supervised Machine Learning Model
### The data is loaded from the NBA API and saved to CSV files for future use in the model
### The data is loaded in three steps: scheduleFrame, gameLogs, and gameLogFeatureSet
### The scheduleFrame contains the schedule for each team for each season
### The gameLogs contains the game logs for each team
### The gameLogFeatureSet contains the features for each game used to train the model

# Import NBA API libraries
from nba_api.stats.static import teams
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam

# Import Pandas for DataFrames
import pandas as pd

# Import NumPy for numerical computing
import numpy as np

# Import JSON for working with JSON data
import json

# Import Difflib for fuzzy string matching
import difflib

# Import Time for tracking cell runtimes
import time

# Import Requests for making HTTP requests
import requests


# Retry wrapper for HTTP requests
def retry(func, retries=10, backoff_factor=1.0):
    ### Inputs: func - function to retry
    ###         retries - number of retries to attempt
    ###         backoff_factor - backoff factor for exponential backoff
    ###
    ### Output: retry_wrapper - wrapper function to retry the function
    """Retry decorator for handling RequestExceptions"""
    
    def retry_wrapper(*args, **kwargs):
        """Wrapper function to retry the function"""
        ### Inputs: *args - positional arguments
        ###         **kwargs - keyword arguments
        ###
        ### Output: func(*args, **kwargs) - function to retry
        
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, json.decoder.JSONDecodeError) as e:
                print(f"Request failed: {e}")
                attempts += 1
                sleep_time = backoff_factor * (2 ** (attempts - 1))
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        raise Exception("Request failed after multiple attempts")
    return retry_wrapper

# Get Season Schedule Function 
def getSeasonScheduleFrame(seasons,seasonType): 
    """Creates a single schedule frame containing the schedule for each team for each season"""
    ### Inputs: seasons - list of seasons to get schedule for
    ###         seasonType - type of season to get schedule for (Regular Season, Playoffs, Pre Season, All Star)
    ###
    ### Output: scheduleFrame - DataFrame containing the schedule for each team for each season
    
    ### CumeStatsTeamGames API Parameters:
    ###   league_id: '00' - NBA
    ###   season: '2020-21' - Season to get schedule for
    ###   season_type_all_star: 'Regular Season' - Type of season to get schedule for (Regular Season, Playoffs, Pre Season, All Star)
    ###   team_id: '1610612737' - ID of team to get schedule for (1610612737, 1610612738, etc.)
    
    ### CumeStatsTeamGames Example API Response:
    ###   Matchup,                          GameID
    ###   04/14/2024 Mavericks at Thunder	0022301196
    
    ### ScheduleFrame Schema we want to create:
    ### Example Columns:    MATCHUP                    , GAME_ID   ,  SEASON,  GAME_DATE , HOME_TEAM_NICKNAME, HOME_TEAM_ID,   AWAY_TEAM_NICKNAME, AWAY_TEAM_ID
    ### Example Rows:       05/16/2021 Rockets at Hawks, 0022001066, 2020-21,  2021-05-16, Hawks             , 1610612737  ,   Rockets           , 1610612745


    # Get GameDate from matchup column: 05/16/2021
    def getGameDate(matchup):
        """Extracts the game date from the matchup column"""
        ### Inputs: matchup - matchup column
        ### 
        ### Output: matchup.partition(' at')[0][:10] - first ten characters of intial partition containing the game date
        
        return matchup.partition(' at')[0][:10] # first ten characters of intial partition

    # Get Home team nickname from matchup column: Hawks
    def getHomeTeam(matchup):
        """Extracts the home team nickname from the matchup column"""
        ### Inputs: matchup - matchup column
        ###
        ### Output: matchup.partition(' at')[2] - last partition of matchup column containing the home team nickname
        
        return matchup.partition(' at')[2] # last partition of matchup column

    # Get Away team nickname from matchup column: Rockets
    def getAwayTeam(matchup):
        """Extracts the away team nickname from the matchup column"""
        ### Inputs: matchup - matchup column
        ###
        ### Output: matchup.partition(' at')[0][10:] - last characters of first partition of matchup column containing the away team nickname
        
        return matchup.partition(' at')[0][10:] # last characters of first partition of matchup column

    # Match nickname from matchup column to the lookup team table to extract the team ID
    def getTeamIDFromNickname(nickname):
        """Extracts the team ID using the team lookup table and the team nickname"""
        ### Inputs: nickname - team nickname
        ###
        ### Output: team ID - ID of the team
        
        return teamLookup.loc[teamLookup['nickname'] == difflib.get_close_matches(nickname,teamLookup['nickname'],1)[0]].values[0][0] 
    
    # Get Regular Season Schedule Function
    @retry
    def getRegularSeasonSchedule(season,teamID,seasonType):
        """Creates a regular season schedule frame for a given team and season"""
        ### Inputs: season - season to get schedule for (2020, 2021, 2022, etc.)
        ###         teamID - ID of team to get schedule for (1610612737, 1610612738, etc.)
        ###         seasonType - type of season to get schedule for (Regular Season, Playoffs, Pre Season, All Star)
        ###
        ### Output: teamGames - DataFrame containing the schedule for a given team and season
        
        # Create schema for season column to match API format
        season = str(season) + "-" + str(season+1)[-2:] # Convert year to season format ie. 2020 -> 2020-21
        
        # Retrieve the team's schedule for the given season in JSON format
        teamGames = cumestatsteamgames.CumeStatsTeamGames(league_id = '00',season = season ,
                                                                      season_type_all_star=seasonType,
                                                                      team_id = teamID).get_normalized_json()

        # Convert JSON to DataFrame
        teamGames = pd.DataFrame(json.loads(teamGames)['CumeStatsTeamGames'])
        
        # Add season column
        teamGames['SEASON'] = season
        
        # Return the DataFrame
        return teamGames    
    
    # Get team lookup table
    teamLookup = pd.DataFrame(teams.get_teams())
    
    # Initialize an empty list to store DataFrames
    scheduleFrames = []  
    
    # Loop through each season and each team to get a schedule
    for season in seasons:
        for id in teamLookup['id']:
            scheduleFrames.append(getRegularSeasonSchedule(season, id, seasonType)) # call getRegularSeasonSchedule function and append to scheduleFrames list
        
        print(f"Loaded schedule for {season} season")
    
    # Concatenate all DataFrames in the list into a single DataFrame
    scheduleFrame = pd.concat(scheduleFrames, ignore_index=True)
    
    # Format the scheduleFrame
    scheduleFrame['GAME_DATE'] = pd.to_datetime(scheduleFrame['MATCHUP'].map(getGameDate)) # convert to datetime
    scheduleFrame['HOME_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getHomeTeam) # create HOME_TEAM_NICKNAME column
    scheduleFrame['HOME_TEAM_ID'] = scheduleFrame['HOME_TEAM_NICKNAME'].map(getTeamIDFromNickname) # create HOME_TEAM_ID column
    scheduleFrame['AWAY_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getAwayTeam) # create AWAY_TEAM_NICKNAME column
    scheduleFrame['AWAY_TEAM_ID'] = scheduleFrame['AWAY_TEAM_NICKNAME'].map(getTeamIDFromNickname) # create AWAY_TEAM_ID column
    
    # Drop unnecessary rows
    scheduleFrame = scheduleFrame.drop_duplicates()
    
    # Reset the indices so there are no gaps
    scheduleFrame = scheduleFrame.reset_index(drop=True)
            
    # Return the scheduleFrame
    return scheduleFrame



# Get Single Game Metrics Function
@retry
def getSingleGameMetrics(gameID,homeTeamID,awayTeamID,awayTeamNickname,seasonYear,gameDate):
    """Creates a DataFrame containing the metrics for a single game"""
    ### Inputs: gameID - ID of the game to get metrics for (0022001066, 0022001067, etc.)
    ###         homeTeamID - ID of the home team (1610612737, 1610612738, etc.)
    ###         awayTeamID - ID of the away team (1610612737, 1610612738, etc.)
    ###         awayTeamNickname - nickname of the away team (Hawks, Rockets, etc.)
    ###         seasonYear - season of the game (2020-21, 2021-22, etc.)
    ###         gameDate - date of the game (2021-05-16, 2021-05-17, etc.)
    ###
    ### Output: data - DataFrame containing the metrics for a single game
    
    ### CumeStatsTeam API Parameters:
    ###   game_ids: '0022001066' - ID of the game to get metrics for
    ###   league_id: '00' - NBA
    ###   season: '2020-21' - Season of the game
    ###   season_type_all_star: 'Regular Season' - Type of season (Regular Season, Playoffs, Pre Season, All Star)
    ###   team_id: '1610612737' - ID of the team to get metrics for (1610612737, 1610612738, etc.)
    
    ### CumeStatsTeam Example API Response:
    ### CITY  ,	NICKNAME ,	TEAM_ID   ,	W,	L,	W_HOME,	L_HOME,	W_ROAD	L_ROAD,	TEAM_TURNOVERS,	TEAM_REBOUNDS,	GP,	GS,	ACTUAL_MINUTES,	ACTUAL_SECONDS, ...
    ### Dallas,	Mavericks,	1610612742,	1,	0,	0     ,	0     ,	1     ,	0     ,	 0	          , 7	         ,  1 ,	5 ,	 240          ,	 0	          , ...

    ### Dataframe Schema we want to create:
    ### Example Columns:    CITY,	NICKNAME,	TEAM_ID,	W,	L,	W_HOME,	L_HOME,	W_ROAD,	L_ROAD,	TEAM_TURNOVERS,	TEAM_REBOUNDS,	GP,	GS,	ACTUAL_MINUTES,	ACTUAL_SECONDS, ...
    ### Example Rows:       Dallas,	Mavericks,	1610612742,	1,	0,	0     ,	0     ,	1     ,	0     ,	 0	          , 7	         ,  1 ,	5 ,	 240          ,	 0	          , ...
    
    # Get Game Stats Function
    @retry
    def getGameStats(teamID, gameID, seasonYear):
        """Creates a DataFrame containing the total team stats for a single game"""
        ### Inputs: teamID - ID of the team to get stats for (1610612737, 1610612738, etc.)
        ###         gameID - ID of the game to get stats for (0022001066, 0022001067, etc.)
        ###         seasonYear - season of the game (2020-21, 2021-22, etc.)
        ###
        ### Output: gameStats - DataFrame containing the total team stats for a single game
        try:
            # Add '00' to the gameID to match the format of the API response
            gameID = '00' + str(gameID)
            gameStats = cumestatsteam.CumeStatsTeam(
                game_ids=gameID, league_id="00",
                season=seasonYear, season_type_all_star="Regular Season",
                team_id=teamID
            ).get_normalized_json()

            # Check for empty responses
            if not gameStats:
                print("Empty response received from the API")
                return None, None
            
            # Convert JSON to DataFrame
            gameStats = json.loads(gameStats)
            
            if not gameStats['TotalTeamStats']:
                print("Empty total team or player response received from the API")
                return None, None
         
            # Return the total team stats DataFrame
            return pd.DataFrame(gameStats['TotalTeamStats'])
        
        # Handle exceptions
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")

        except Exception as e:
            print(f"Error fetching game stats for teamID: {teamID}, gameID: {gameID}, seasonYear: {seasonYear}")
            print(f"Exception: {e}")
            
    # Get Game Stats for Home Team
    data = getGameStats(homeTeamID,gameID,seasonYear)

    # Handle empty responses
    if data is None:
        return None

    # Fill in the away team data    
    data.at[1,'NICKNAME'] = awayTeamNickname # away team nickname
    data.at[1,'TEAM_ID'] = awayTeamID # away team ID
    
    # Away team stats
    data.at[1,'OFFENSIVE_EFFICIENCY'] = (data.at[1,'FG'] + data.at[1,'AST'])/(data.at[1,'FGA'] - data.at[1,'OFF_REB'] + data.at[1,'AST'] + data.at[1,'TOTAL_TURNOVERS'])
    data.at[1,'SCORING_MARGIN'] = data.at[1,'PTS'] - data.at[0,'PTS']
 
    # Home team stats
    data.at[0,'OFFENSIVE_EFFICIENCY'] = (data.at[0,'FG'] + data.at[0,'AST'])/(data.at[0,'FGA'] - data.at[0,'OFF_REB'] + data.at[0,'AST'] + data.at[0,'TOTAL_TURNOVERS'])
    data.at[0,'SCORING_MARGIN'] = data.at[0,'PTS'] - data.at[1,'PTS']
        
    # Add additional columns
    data['SEASON'] = seasonYear
    data['GAME_DATE'] = gameDate
    data['GAME_ID'] = gameID

    # Return the DataFrame
    return data


# Get Game Logs Function
def getGameLogs(gameLogs, scheduleFrame):
    """Creates a DataFrame containing the game logs for each team used by GameLogFeatureSet"""
    ### Inputs: gameLogs - DataFrame containing the game logs for each team
    ###         scheduleFrame - DataFrame containing the schedule for each team for each season
    ###
    ### Output: gameLogs - DataFrame containing the game logs for each team
    
    ### Example Columns:    CITY,	NICKNAME,	TEAM_ID,	W,	L,	W_HOME,	L_HOME,	W_ROAD,	L_ROAD,	TEAM_TURNOVERS,	TEAM_REBOUNDS,	GP,	GS,	ACTUAL_MINUTES,	ACTUAL_SECONDS, ...
    ### Example Rows:       Dallas,	Mavericks,	1610612742,	1,	0,	0     ,	0     ,	1     ,	0     ,	 0	          , 7	         ,  1 ,	5 ,	 240          ,	 0	          , ...
    
    ### Additional Columns: HOME_FLAG, AWAY_FLAG, TOTAL_GAMES_PLAYED, TOTAL_WINS, TOTAL_WIN_PCTG, HOME_GAMES_PLAYED, HOME_WINS, HOME_WIN_PCTG, AWAY_GAMES_PLAYED, AWAY_WINS, AWAY_WIN_PCTG, ROLLING_OE, ROLLING_SCORING_MARGIN, NUM_REST_DAYS
    ### Additional Rows:    1        , 0        , 1               , 1         , 1.0           , 1                , 1        , 1.0           , 0                , 0        , 0.0           , 1.0        , 0.0                   , 0
    
    
    ### Functions to prepare additional columns after gameLogs table loads
    
    # Sets flag to 1 if game was played at home and there was a corresponding win or loss
    def getHomeAwayFlag(gameDF):
        """Sets a flag to 1 if the game was played at home and there was a corresponding win or loss"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the HOME_FLAG and AWAY_FLAG columns added
        
        gameDF['HOME_FLAG'] = np.where((gameDF['W_HOME']==1) | (gameDF['L_HOME']==1),1,0)
        gameDF['AWAY_FLAG'] = np.where((gameDF['W_ROAD']==1) | (gameDF['L_ROAD']==1),1,0)
        return gameDF
    
    # Gets the total number of games and counts the number of wins for each team up to that point to calculate win percentage
    def getTotalWinPctg(gameDF):
        """Gets the total win percentage for each team"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the TOTAL_GAMES_PLAYED, TOTAL_WINS, and TOTAL_WIN_PCTG columns added
        
        gameDF['TOTAL_GAMES_PLAYED'] = gameDF.groupby(['TEAM_ID','SEASON'])['GAME_DATE'].rank(ascending=True)
        gameDF['TOTAL_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W'].cumsum()
        gameDF['TOTAL_WIN_PCTG'] = gameDF['TOTAL_WINS']/gameDF['TOTAL_GAMES_PLAYED']
        return gameDF.drop(['TOTAL_GAMES_PLAYED','TOTAL_WINS'],axis=1)

    # Gets the home win percentage for each team
    def getHomeWinPctg(gameDF):
        """Gets the home win percentage for each team"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the HOME_GAMES_PLAYED, HOME_WINS, and HOME_WIN_PCTG columns added
        
        gameDF['HOME_GAMES_PLAYED'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['HOME_FLAG'].cumsum()
        gameDF['HOME_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W_HOME'].cumsum()
        gameDF['HOME_WIN_PCTG'] = gameDF['HOME_WINS']/gameDF['HOME_GAMES_PLAYED']
        return gameDF.drop(['HOME_GAMES_PLAYED','HOME_WINS'],axis=1)

    # Gets the away win percentage for each team
    def getAwayWinPctg(gameDF):
        """Gets the away win percentage for each team"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the AWAY_GAMES_PLAYED, AWAY_WINS, and AWAY_WIN_PCTG columns added
        
        gameDF['AWAY_GAMES_PLAYED'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['AWAY_FLAG'].cumsum()
        gameDF['AWAY_WINS'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['W_ROAD'].cumsum()
        gameDF['AWAY_WIN_PCTG'] = gameDF['AWAY_WINS']/gameDF['AWAY_GAMES_PLAYED']
        return gameDF.drop(['AWAY_GAMES_PLAYED','AWAY_WINS'],axis=1)

    # Gets the rolling average offensive efficiency for each team taking average of last 3 games
    def getRollingOE(gameDF):
        """Gets the rolling average offensive efficiency for each team taking average of last 3 games"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the ROLLING_OE column added
        
        gameDF['ROLLING_OE'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['OFFENSIVE_EFFICIENCY'].transform(lambda x: x.rolling(3, 1).mean())
        return gameDF
    
    # Gets the rolling average scoring margin for each team taking average of last 3 games
    def getRollingScoringMargin(gameDF):
        """Gets the rolling average scoring margin for each team taking average of last 3 games"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the ROLLING_SCORING_MARGIN column added
        
        gameDF['ROLLING_SCORING_MARGIN'] = gameDF.sort_values(by='GAME_DATE').groupby(['TEAM_ID','SEASON'])['SCORING_MARGIN'].transform(lambda x: x.rolling(3, 1).mean())
        return gameDF
    
    # Gets the current game data, shifts it back one, then caluclates the time inbetween in days floating point
    def getRestDays(gameDF):
        """Calculates the number of rest days between games for each team"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team with the NUM_REST_DAYS column added
        
        # Ensure GAME_DATE is in datetime format
        gameDF['GAME_DATE'] = pd.to_datetime(gameDF['GAME_DATE'], errors='coerce')

        # Calculate LAST_GAME_DATE by shifting GAME_DATE
        gameDF['LAST_GAME_DATE'] = gameDF.sort_values(by='GAME_DATE') \
                                        .groupby(['TEAM_ID', 'SEASON'])['GAME_DATE'].shift(1)
        
        # Calculate the number of rest days
        gameDF['NUM_REST_DAYS'] = (gameDF['GAME_DATE'] - gameDF['LAST_GAME_DATE']).dt.days

        # Drop LAST_GAME_DATE before returning so just the number of rest days is left
        return gameDF.drop('LAST_GAME_DATE', axis=1)
    
    
    # Get Game Logs Function
    start = time.perf_counter_ns() # Track cell's runtime
    
    # Captures only the home team data    
    i = int(len(gameLogs)/2)
    
    # Initialize a list to store the DataFrames
    gameLogs_list = [gameLogs]

    while i < len(scheduleFrame):
        time.sleep(1)  # Sleep for 1 second to avoid API rate limit
        
        # Get the new game log
        new_game_log = getSingleGameMetrics(
            scheduleFrame.at[i, 'GAME_ID'],
            scheduleFrame.at[i, 'HOME_TEAM_ID'],
            scheduleFrame.at[i, 'AWAY_TEAM_ID'],
            scheduleFrame.at[i, 'AWAY_TEAM_NICKNAME'],
            scheduleFrame.at[i, 'SEASON'],
            scheduleFrame.at[i, 'GAME_DATE']
        )
        
        # Handle empty responses
        if new_game_log is None:
            i += 1
            continue
        
        # Append the new DataFrame to the list
        gameLogs_list.append(new_game_log)
        
        i += 1  # Increment the index
        
        # Track the time it took to load x amount of records
        end = time.perf_counter_ns()
        
        # Output time it took to load x amount of records
        if i % 100 == 0:
            mins = ((end - start) / 1e9) / 60
            print(f"Loaded {i} records in {mins:.2f} minutes")
            
    
    # Concatenate all DataFrames in the list into a single DataFrame
    gameLogs = pd.concat(gameLogs_list, ignore_index=True)
    
    # Add additional columns to the gameLogs DataFrame
    gameLogs = getHomeAwayFlag(gameLogs)
    gameLogs = getHomeWinPctg(gameLogs)
    gameLogs = getAwayWinPctg(gameLogs)
    gameLogs = getTotalWinPctg(gameLogs)
    gameLogs = getRollingScoringMargin(gameLogs)
    gameLogs = getRollingOE(gameLogs)
    gameLogs = getRestDays(gameLogs)

    # Drop unnecessary columns, reset the index, and return the DataFrame containing the game logs for every game in every season
    return gameLogs.reset_index(drop=True)


# Get Game Log Feature Set Function to use to train the model(s)
def getGameLogFeatureSet(gameDF):
    """Creates a DataFrame extracting the feature set for every gamelog in the gameDF dropping unnecessary columns"""
    ### Inputs: gameDF - DataFrame containing the game logs for each team for each season
    ###
    ### Output: gameLogFeatureSet - DataFrame containing the feature for each game
    
    ### Example Columns:    HOME_LAST_GAME_OE, HOME_LAST_GAME_HOME_WIN_PCTG, HOME_NUM_REST_DAYS, HOME_LAST_GAME_AWAY_WIN_PCTG, HOME_LAST_GAME_TOTAL_WIN_PCTG, HOME_LAST_GAME_ROLLING_SCORING_MARGIN, HOME_LAST_GAME_ROLLING_OE, HOME_W, HOME_TEAM_ID, HOME_GAME_ID, HOME_SEASON, AWAY_LAST_GAME_OE, AWAY_LAST_GAME_HOME_WIN_PCTG, AWAY_NUM_REST_DAYS, AWAY_LAST_GAME_AWAY_WIN_PCTG, AWAY_LAST_GAME_TOTAL_WIN_PCTG, AWAY_LAST_GAME_ROLLING_SCORING_MARGIN, AWAY_LAST_GAME_ROLLING_OE, AWAY_TEAM_ID, AWAY_GAME_ID, AWAY_SEASON
    ### Example Rows:       0.0              , 1.0                         , 0                 , 0.0                         , 0.0                          , 0.0                                  , 0.0                      , 1  ,     1610612737 ,  0022001066 ,     2020-21, 0.0              ,   0.0                       ,   0               ,  1.0                        ,    1.0                       ,        0.0                           ,    0.0                   ,   1610612745,   0022001066,  2020-21

    # Allows for the shifting of game log records to get the previous game's offensive efficiency, home win percentage, away win percentage, total win percentage, rolling scoring margin, and rolling offensive efficiency
    def shiftGameLogRecords(gameDF):
        """Shifts the game log records to get the previous game's offensive efficiency, home win percentage, away win percentage, total win percentage, rolling scoring margin, and rolling offensive efficiency"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team for each season
        ###
        ### Output: gameDF - DataFrame containing the game logs for each team for each season with the shifted columns added (next game records as new rows)
        
        gameDF['LAST_GAME_OE'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['OFFENSIVE_EFFICIENCY'].shift(1)
        gameDF['LAST_GAME_HOME_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['HOME_WIN_PCTG'].shift(1)
        gameDF['LAST_GAME_AWAY_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['AWAY_WIN_PCTG'].shift(1)
        gameDF['LAST_GAME_TOTAL_WIN_PCTG'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['TOTAL_WIN_PCTG'].shift(1)
        gameDF['LAST_GAME_ROLLING_SCORING_MARGIN'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['ROLLING_SCORING_MARGIN'].shift(1)
        gameDF['LAST_GAME_ROLLING_OE'] = gameDF.sort_values('GAME_DATE').groupby(['TEAM_ID','SEASON'])['ROLLING_OE'].shift(1)
    
    # Creates a home team frame
    def getHomeTeamFrame(gameDF):
        """Extracts the home team frame from the gameDF"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team for each season
        ###
        ### Output: homeTeamFrame - DataFrame containing the home team frame
        
        homeTeamFrame = gameDF[gameDF['CITY'] != 'OPPONENTS']
        homeTeamFrame = homeTeamFrame[['LAST_GAME_OE','LAST_GAME_HOME_WIN_PCTG','NUM_REST_DAYS','LAST_GAME_AWAY_WIN_PCTG','LAST_GAME_TOTAL_WIN_PCTG','LAST_GAME_ROLLING_SCORING_MARGIN','LAST_GAME_ROLLING_OE','W','TEAM_ID','GAME_ID','SEASON']]

        # Adds HOME_ prefix to all columns except GAME_ID and SEASON
        colRenameDict = {}
        for col in homeTeamFrame.columns:
            if (col != 'GAME_ID') & (col != 'SEASON') :
                colRenameDict[col] = 'HOME_' + col 

        # Rename the columns
        homeTeamFrame.rename(columns=colRenameDict,inplace=True)

        # Return the home team frame
        return homeTeamFrame

    # Creates an away team frame
    def getAwayTeamFrame(gameDF):
        """Extracts the away team frame from the gameDF"""
        ### Inputs: gameDF - DataFrame containing the game logs for each team for each season
        ###
        ### Output: awayTeamFrame - DataFrame containing the away team frame
        
        # Strip whitespace from CITY column
        gameDF['CITY'] = gameDF['CITY'].str.strip()    
        
        # Get the away team frame
        awayTeamFrame = gameDF[gameDF['CITY'] == 'OPPONENTS']
        awayTeamFrame = awayTeamFrame[['LAST_GAME_OE','LAST_GAME_HOME_WIN_PCTG','NUM_REST_DAYS','LAST_GAME_AWAY_WIN_PCTG','LAST_GAME_TOTAL_WIN_PCTG','LAST_GAME_ROLLING_SCORING_MARGIN','LAST_GAME_ROLLING_OE','TEAM_ID','GAME_ID','SEASON']]

        # Adds AWAY_ prefix to all columns except GAME_ID and SEASON
        colRenameDict = {}
        for col in awayTeamFrame.columns:
            if (col != 'GAME_ID') & (col != 'SEASON'):
                colRenameDict[col] = 'AWAY_' + col 

        # Rename the columns
        awayTeamFrame.rename(columns=colRenameDict,inplace=True)

        # Return the away team frame
        return awayTeamFrame
    
    # Shift the game log records
    shiftGameLogRecords(gameDF)
    
    # Create the home and away team frames
    awayTeamFrame = getAwayTeamFrame(gameDF)
    homeTeamFrame = getHomeTeamFrame(gameDF)
    
    # Merge the home and away team frames on GAME_ID and SEASON to create a single row for each game
    return pd.merge(homeTeamFrame, awayTeamFrame, how="inner", on=[ "GAME_ID","SEASON"]) # keep common keys for later
    


def initializeDataSets(seasons = [2020, 2021, 2022, 2023, 2024], seasonType = 'Regular Season'):
    """Initializes the scheduleFrame, gameLogs, and gameLogFeatureSet DataFrames"""
    ### Inputs: seasons - list of seasons to get schedule for
    ###         seasonType - type of season to get schedule for (Regular Season, Playoffs, Pre Season, All Star)
    ###
    ### Output: scheduleFrame.csv, gameLogs.csv, nbaHomeWinLossModelDataset.csv

    # Create a new scheduleFrame
    print("Loading scheduleFrame...")
    scheduleFrame = pd.DataFrame()
    scheduleFrame = getSeasonScheduleFrame(seasons,seasonType)
        
    # Save the scheduleFrame to CSV
    scheduleFrame.to_csv('scheduleFrame.csv', index=False)
    print("Saved scheduleFrame to CSV.")
    
        
    # Create a new gameLogsFrame
    print("Loading gameLogs...")
    gameLogs = pd.DataFrame()
    gameLogs = getGameLogs(gameLogs, scheduleFrame)

    # Save the gameLogs to CSV
    gameLogs.to_csv('gameLogs.csv', index=False)
    print("Saved gameLogs to CSV.")
    
    
    # Create a new gameLogFeatureSet
    print("Loading gameLogFeatureSet...")
    gameLogFeatureSet  = pd.DataFrame()
    gameLogFeatureSet = getGameLogFeatureSet(gameLogs)
    
    # Save the gameLogFeatureSet to CSV
    gameLogFeatureSet.to_csv('nbaHomeWinLossModelDataset.csv', index=False)
    print("Saved nbaHomeWinLossModelDataset to CSV.")


def reload_schedule_frame(filename, seasons = [2020, 2021, 2022, 2023, 2024], seasonType = 'Regular Season'):
    """Reloads the scheduleFrame DataFrame and saves it to a new CSV file"""
    ### Inputs: filename - filename to save new scheduleFrame to
    ###         seasons - list of seasons to get schedule for
    ###         seasonType - type of season to get schedule for (Regular Season, Playoffs, Pre Season, All Star)
    ###
    ### Output: filename.csv - new scheduleFrame saved to CSV
    
    # Create a new scheduleFrame
    print("Loading scheduleFrame...")
    scheduleFrame = getSeasonScheduleFrame(seasons,seasonType)
    scheduleFrame.to_csv(filename, index=False)
    print(f"Saved new ", {filename} ," to CSV.")


def reload_game_logs(scheduleFrame_filename, gameLogs_filename):
    """Reloads the gameLogs DataFrame and saves it to a new CSV file"""
    ### Inputs: scheduleFrame_filename - filename of scheduleFrame to load
    ###         gameLogs_filename - filename to save new gameLogs to
    ###
    ### Output: gameLogs_filename.csv - new gameLogs saved to CSV
    
    # Create a new scheduleFrame for current season only!
    scheduleFrame = pd.read_csv(scheduleFrame_filename)
    scheduleFrame = scheduleFrame[scheduleFrame['SEASON'] == '2024-25']
    scheduleFrame = scheduleFrame.reset_index(drop=True)
    
    # Create a new gameLogs for current season only!
    print("Loading gameLogs...")
    gameLogs = pd.DataFrame()
    gameLogs = getGameLogs(gameLogs, scheduleFrame)
    gameLogs.to_csv(gameLogs_filename, index=False)
    print(f"Saved new ", {gameLogs_filename} ," to CSV.")
    

def reload_feature_set(gameLogs_filename, gameLogFeatureSet_filename):
    """Reloads the gameLogFeatureSet DataFrame and saves it to a new CSV file"""
    ### Inputs: gameLogs_filename - filename of gameLogs to load
    ###         gameLogFeatureSet_filename - filename to save new gameLogFeatureSet to
    ###
    ### Output: gameLogFeatureSet_filename.csv - new gameLogFeatureSet saved to CSV
    
    # Create a new gameLogFeatureSet
    gameLogs = pd.read_csv(gameLogs_filename)
    gameLogFeatureSet = getGameLogFeatureSet(gameLogs)
    
    # Save the gameLogFeatureSet to CSV
    gameLogFeatureSet.to_csv(gameLogFeatureSet_filename, index=False)
    print(f"Saved new ", {gameLogFeatureSet_filename} ," to CSV.")
    
    
    
if __name__ == "__main__":
    """Main function to initialize or reload the data sets"""
    
    # Set to True to initialize data sets, False to reload data sets
    initialize_data_sets = False
    
    if initialize_data_sets:    
        # Initialize Data Sets
        initializeDataSets(seasons=[2024], seasonType='Regular Season')
    
    else:
        # Reload Schedule Frame
        filename_schedule = 'scheduleFrame.csv' # filename to save new scheduleFrame to
        reload_schedule_frame(filename_schedule, seasons=[2024], seasonType='Regular Season')
        
        # Reload Game Logs
        filename_game_logs = 'gameLogs.csv' # filename to save new gameLogs to
        reload_game_logs(filename_schedule, filename_game_logs)
        
        # Reload Game Log Feature Set
        filename_feature_set = 'nbaHomeWinLossModelDataset.csv' # filename to save new gameLogFeatureSet to
        reload_feature_set(filename_game_logs, filename_feature_set)
        