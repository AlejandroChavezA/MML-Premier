#!/usr/bin/env python3
"""
Predict v2
==========
Script para hacer predicciones con el modelo entrenado.

Uso:
    python src_v2/predict.py                    # Predicción interactiva
    python src_v2/predict.py --test          # Test con partidos conocidos
    python src_v2/predict.py --jornada 30  # Predecir jornada 30
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from season_utils import get_current_season

from src_v2.features.feature_engineer import get_feature_engineer
from src_v2.models.winner_predictor import get_winner_predictor
from src_v2.models.goals_predictor import get_goals_predictor


def predict_match(home: str, away: str, date=None):
    """Predecir un partido"""
    print(f"\n⚽ {home} vs {away}")
    print("-" * 40)
    
    # Load models
    fe = get_feature_engineer("data/cleaned")
    fe.load_data()
    
    winner = get_winner_predictor("models")
    if not winner.load():
        print("❌ Error: modelo no encontrado. Ejecuta primero python src_v2/train.py")
        return None
    
    goals = get_goals_predictor("models")
    goals.load()
    
    if date is None:
        date = datetime.now()
    
    # Winner prediction
    pred = winner.predict(home, away, date, fe)
    
    if 'error' in pred:
        print(f"❌ Error: {pred['error']}")
        return None
    
    print(f"  Predicción: {pred['predicted']}")
    print(f"  Confianza: {pred['confidence']:.0%}")
    print(f"  Probabilidades:")
    for k, v in pred['probabilities'].items():
        print(f"    {k}: {v:.0%}")
    
    # Goals prediction
    goals_pred = goals.predict(home, away, date, fe)
    
    if 'error' not in goals_pred:
        print(f"\n  Goles esperados: {goals_pred['expected_goals']:.1f}")
        print(f"  Over/Under 2.5:")
        ou = goals_pred['markets']['over_2.5']
        print(f"    {ou['prediction']} ({ou['over_prob']:.0%})")
    
    return pred


def test_with_known_results():
    """Test con partidos cuyos resultados ya conocemos"""
    print("\n🧪 TEST CON RESULTADOS CONOCIDOS")
    print("=" * 50)
    
    # Load prediction history
    history_path = PROJECT_ROOT / "data" / "prediction_history.json"
    
    if not history_path.exists():
        print("❌ No hay historial")
        return
    
    import json
    with open(history_path) as f:
        history = json.load(f)
    
    # Get last 10 matches with results
    recent = sorted(history, key=lambda x: x['match_date'])[-10:]
    
    correct = 0
    total = 0
    
    fe = get_feature_engineer("data/cleaned")
    fe.load_data()
    
    winner = get_winner_predictor("models")
    winner.load()
    
    print("\nPredicciones vs Resultados:")
    for p in recent:
        home = p['home_team']
        away = p['away_team']
        
        # Predict
        date = datetime.fromisoformat(p['match_date'])
        pred = winner.predict(home, away, date, fe)
        
        if 'error' in pred:
            continue
        
        # Compare with actual
        actual = 'LOCAL' if p['home_score'] > p['away_score'] else (
            'VISITANTE' if p['home_score'] < p['away_score'] else 'EMPATE'
        )
        
        correct_flag = pred['predicted'] == actual
        if correct_flag:
            correct += 1
        total += 1
        
        status = "✅" if correct_flag else "❌"
        print(f"  {status} {pred['predicted']:10} vs {actual:10} | {home} vs {away}")
    
    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.0f}%)")


def predict_jornada(matchday: int):
    """Predecir todos los partidos de una jornada"""
    print(f"\n📅 JORNADA {matchday}")
    print("=" * 50)
    
    # Load matches (temporada actual)
    matches = pd.read_csv(f"data/cleaned/matches_{get_current_season()}_cleaned.csv")
    matches['date'] = pd.to_datetime(matches['date'], utc=True).dt.tz_localize(None)
    
    week_matches = matches[matches['matchday'] == matchday]
    
    if len(week_matches) == 0:
        print(f"No hay partidos para jornada {matchday}")
        return
    
    fe = get_feature_engineer("data/cleaned")
    fe.load_data()
    
    winner = get_winner_predictor("models")
    winner.load()
    
    print(f"\n{len(week_matches)} partidos:")
    for _, m in week_matches.iterrows():
        if m['status'] == 'TIMED':  # Only upcoming
            date = m['date']
            pred = winner.predict(m['home_team'], m['away_team'], date, fe)
            
            if 'error' not in pred:
                conf = pred['confidence']
                marker = "🔥" if conf > 0.5 else "  "
                print(f"  {marker} {pred['predicted']:10} ({conf:.0%}) | {m['home_team']} vs {m['away_team']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Predictor v2")
    parser.add_argument('--test', action='store_true', help='Test con resultados conocidos')
    parser.add_argument('--jornada', type=int, help='Predecir jornada N')
    parser.add_argument('--home', type=str, help='Equipo local')
    parser.add_argument('--away', type=str, help='Equipo visitante')
    parser.add_argument('--date', type=str, help='Fecha (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.test:
        test_with_known_results()
        return
    
    if args.jornada:
        predict_jornada(args.jornada)
        return
    
    if args.home and args.away:
        date = None
        if args.date:
            date = datetime.fromisoformat(args.date)
        predict_match(args.home, args.away, date)
        return
    
    # Interactive mode
    print("⚽ PREDICTOR PREMIER LEAGUE v2")
    print("=" * 40)
    print("\nUso:")
    print("  python src_v2/predict.py --test              # Test con resultados conocidos")
    print("  python src_v2/predict.py --jornada 30         # Predecir jornada 30")
    print("  python src_v2/predict.py --home 'Arsenal FC' --away 'Liverpool FC'")
    print("\nO interactivo:")
    print("  python src_v2/predict.py --home 'Chelsea FC' --away 'Manchester City FC'")


if __name__ == "__main__":
    main()