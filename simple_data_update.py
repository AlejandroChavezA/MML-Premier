#!/usr/bin/env python3

import json
import csv
import subprocess
from datetime import datetime
import os

def download_premier_league_data():
    """Download Premier League data using curl and process with Python"""
    print("🔄 DOWNLOADING PREMIER LEAGUE DATA")
    print("=" * 50)
    
    # API details
    api_token = "fd9ecc768e3644dfa9b30e9536031700"
    season = 2025
    
    try:
        # Download matches using curl
        print("⚽ Downloading matches...")
        curl_command = [
            "curl", "-H", f"X-Auth-Token: {api_token}",
            f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
        ]
        
        result = subprocess.run(curl_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error downloading data: {result.stderr}")
            return False
            
        data = json.loads(result.stdout)
        matches = data['matches']
        
        print(f"✅ Downloaded {len(matches)} matches")
        
        # Process matches
        processed_matches = []
        finished_count = 0
        
        for match in matches:
            match_info = {
                'id': match['id'],
                'date': match['utcDate'],
                'matchday': match['matchday'],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'status': match['status'].upper()
            }

            if match['status'].upper() == 'FINISHED':
                match_info['home_score'] = match['score']['fullTime']['home']
                match_info['away_score'] = match['score']['fullTime']['away']
                finished_count += 1
            else:
                match_info['home_score'] = ''
                match_info['away_score'] = ''
            
            # Calculate additional fields
            if match['status'] == 'FINISHED':
                total_goals = match_info['home_score'] + match_info['away_score']
                goal_difference = match_info['home_score'] - match_info['away_score']
                
                if goal_difference > 0:
                    result = 'LOCAL'
                elif goal_difference < 0:
                    result = 'VISITANTE'
                else:
                    result = 'EMPATE'
            else:
                total_goals = 0.0
                goal_difference = 0.0
                result = 'SIN JUGAR'
            
            match_info.update({
                'total_goals': total_goals,
                'goal_difference': goal_difference,
                'result': result
            })
            
            processed_matches.append(match_info)
        
        # Save to data directory
        os.makedirs('data', exist_ok=True)
        
        # Save raw data
        with open('data/matches_2025_updated.csv', 'w', newline='') as csvfile:
            fieldnames = ['id', 'date', 'matchday', 'home_team', 'away_team', 
                         'home_score', 'away_score', 'status', 'total_goals', 
                         'goal_difference', 'result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_matches)
        
        # Also save to cleaned directory to update existing data
        os.makedirs('data/cleaned', exist_ok=True)
        with open('data/cleaned/matches_2025_cleaned.csv', 'w', newline='') as csvfile:
            fieldnames = ['id', 'date', 'matchday', 'home_team', 'away_team', 
                         'home_score', 'away_score', 'status', 'total_goals', 
                         'goal_difference', 'result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_matches)
        
        print(f"📊 RESULTS:")
        print(f"   Total matches: {len(matches)}")
        print(f"   Finished: {finished_count}")
        print(f"   Scheduled: {len(matches) - finished_count}")
        print(f"   New matches: {finished_count - 220}")  # Previous was 220
        
        # Show recent finished matches
        recent_matches = [m for m in processed_matches if m['status'] == 'FINISHED' and '2026-02' in m['date']]
        if recent_matches:
            print(f"\n🏆 RECENT FINISHED MATCHES:")
            for match in recent_matches[:5]:
                print(f"   {match['date'][:10]}: {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}")
        
        print(f"\n✅ DATA UPDATED SUCCESSFULLY!")
        print(f"📁 Saved to: data/matches_2025_updated.csv")
        print(f"📁 Updated: data/cleaned/matches_2025_cleaned.csv")
        print(f"\n🚀 Next steps:")
        print(f"   python3 main.py --train  # Retrain models with new data")
        print(f"   python3 main.py           # Start prediction menu")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    download_premier_league_data()