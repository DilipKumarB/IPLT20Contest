import pandas as pd

with open('.//ipl_csv2//all_matches.csv') as f:
    ipl_data = pd.read_csv(f)

relevantColumns = ['match_id', 'venue', 'innings', 'ball',
                   'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
                   'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
                   'penalty']

ipl_data = ipl_data[relevantColumns]

ipl_data['total_runs'] = ipl_data['runs_off_bat'] + ipl_data['extras']

ipl_data = ipl_data.drop(columns=['runs_off_bat','extras'])

ipl_data = ipl_data[ipl_data['ball']<=5.6]

ipl_data = ipl_data[ipl_data['innings']<=2]

ipl_data = ipl_data.groupby(['match_id',
                             'venue',
                             'innings',
                             'batting_team',
                             'bowling_team']).total_runs.sum()

ipl_data = ipl_data.reset_index()

ipl_data = ipl_data.drop(columns=['match_id'])

ipl_data.to_csv('myPreprocessed.csv',index=False)
