"""
Model Evaluator
==============
Evalúa rendimiento de modelos y analiza predicciones.

Dependencias:
- pandas, sklearn.metrics
- json (prediction history)

Usa:
- models.winner_predictor
- models.goals_predictor
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json


class ModelEvaluator:
    """Evalúa modelos de predicción"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
    
    def evaluate_recent_predictions(self, n_matches: int = 20) -> Dict:
        """Evaluar últimas N predicciones desde history"""
        history_path = self.data_dir / "prediction_history.json"
        
        if not history_path.exists():
            return {'error': 'No se encontró prediction_history.json'}
        
        with open(history_path) as f:
            history = json.load(f)
        
        # Get most recent matches
        recent = sorted(history, key=lambda x: x['match_date'])[-n_matches:]
        
        if not recent:
            return {'error': 'No hay predicciones'}
        
        correct = sum(1 for p in recent if p.get('1x2_correct', False))
        total = len(recent)
        
        # By result type
        by_type = {'LOCAL': {'correct': 0, 'total': 0},
                  'EMPATE': {'correct': 0, 'total': 0},
                  'VISITANTE': {'correct': 0, 'total': 0}}
        
        for p in recent:
            pred = p.get('predicted_1x2', '')
            if pred in by_type:
                by_type[pred]['total'] += 1
                if p.get('1x2_correct'):
                    by_type[pred]['correct'] += 1
        
        return {
            'n_matches': total,
            'correct': correct,
            'accuracy': correct / total * 100,
            'by_type': by_type,
        }
    
    def evaluate_time_window(self, days: int = 14) -> Dict:
        """Evaluar predicciones de últimos N días"""
        history_path = self.data_dir / "prediction_history.json"
        
        if not history_path.exists():
            return {'error': 'No se encontró prediction_history.json'}
        
        with open(history_path) as f:
            history = json.load(f)
        
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = [
            p for p in history 
            if datetime.fromisoformat(p['match_date']) > cutoff
        ]
        
        if not recent:
            return {'n_matches': 0, 'accuracy': 0}
        
        correct = sum(1 for p in recent if p.get('1x2_correct', False))
        
        return {
            'n_matches': len(recent),
            'correct': correct,
            'accuracy': correct / len(recent) * 100,
            'days': days,
        }
    
    def check_overfitting(self, train_acc: float, test_acc: float, 
                        cv_mean: float) -> Dict:
        """Detectar overfitting"""
        gap = train_acc - test_acc
        
        status = "OK"
        if gap > 0.3:
            status = "SEVERE_OVERFIT"
        elif gap > 0.15:
            status = "MODERATE_OVERFIT"
        
        recommendations = []
        if gap > 0.15:
            recommendations.append("Reducir max_depth")
            recommendations.append("Aumentar min_samples_split")
            recommendations.append("Usar más regularización")
        
        if abs(cv_mean - test_acc) > 0.1:
            recommendations.append("Verificar data leak (time-series split)")
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': cv_mean,
            'gap': gap,
            'status': status,
            'recommendations': recommendations,
        }
    
    def print_summary(self):
        """Imprimir resumen de evaluación"""
        print("\n📊 EVALUACIÓN DE MODELOS")
        print("=" * 50)
        
        # Recent performance
        recent = self.evaluate_time_window(14)
        print(f"\nÚltimas 2 semanas:")
        print(f"  Partidos: {recent.get('n_matches', 0)}")
        print(f"  Accuracy: {recent.get('accuracy', 0):.1f}%")
        
        # By prediction type
        recent_full = self.evaluate_recent_predictions(20)
        if 'by_type' in recent_full:
            print(f"\nPor tipo de predicción (últimos 20):")
            for ptype, stats in recent_full['by_type'].items():
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    print(f"  {ptype}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")


def get_evaluator(data_dir: str = "data", 
                models_dir: str = "models") -> ModelEvaluator:
    """Factory function"""
    return ModelEvaluator(data_dir, models_dir)