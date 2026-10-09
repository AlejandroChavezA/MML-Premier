"""Menú interactivo para Liga MX -- paralelo a menu_interface.py::PredictionMenu
(Premier, legacy) pero construido sobre el ensamble de core/ en vez del predictor
sklearn legacy, que Liga MX no tiene (has_legacy_baseline: false en
config/leagues/liga_mx.yaml).

Sin "cambiar modelo" ni "rendimiento de modelos": el ensamble de core/ no es un
menú de modelos intercambiables (random_forest vs gradient_boosting vs...), son
3 modelos complementarios (OutcomeXGB + DixonColes + GoalsGLM) que siempre
corren juntos y se reconcilian en core/ensemble.py::reconcile() -- no hay "mejor
modelo" que elegir, así que esa opción no tiene equivalente real acá.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import menu_actions  # noqa: E402
from menu_interface import PredictionMenu, TEAM_METADATA  # noqa: E402
from season_utils import get_current_phase, get_current_season  # noqa: E402

from core.league_config import load_league_config  # noqa: E402
from core.panel_export import ensemble_predictions_to_panel_format  # noqa: E402
from core.predict_match import predict_match_ensemble  # noqa: E402
from core.predict_week import load_ensemble_models, predict_week  # noqa: E402
from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402


class LigaMXMenu:
    def __init__(self):
        self.cfg = load_league_config("liga_mx")
        self.fe: Optional[FeatureEngineer] = None
        self.outcome_model = None
        self.dc_model = None
        self.glm_model = None
        self.team_codes, self.team_logos = TEAM_METADATA["liga_mx"]
        # Instancia "vacía" (sin .initialize()) solo como toolbox: get_team_code/
        # get_team_full_name/get_api_key/_send_predictions_to_panel ya son
        # league-agnostic o no dependen de self.predictor -- mismo patrón que
        # run.py::_cmd_jornada_multileague, para no duplicar esa lógica acá.
        self._panel = PredictionMenu()

    def initialize(self) -> bool:
        print("Inicializando Liga MX...")
        self.fe = FeatureEngineer(str(self.cfg.data_dir_path), extra_features=self.cfg.extra_features)
        if not self.fe.load_data():
            return False
        try:
            self.outcome_model, self.dc_model, self.glm_model = load_ensemble_models(self.cfg)
        except RuntimeError as e:
            print(f"ERROR {e}")
            return False
        print("OK Sistema Liga MX inicializado correctamente")
        return True

    def _season_phase(self):
        season = get_current_season(league_cfg=self.cfg)
        phase = get_current_phase(self.cfg)
        return season, phase

    def _code(self, team: str) -> str:
        return self._panel.get_team_code(team, league="liga_mx")

    # ------------------------------------------------------------------ menú

    def display_main_menu(self):
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print("=" * 50)
            print("  PREDICTOR LIGA MX")
            print("=" * 50)
            print("1. Predicción de jornada completa")
            print("2. Predicción por jornada (detalles + envío a dashboard)")
            print("3. Predicción partido por partido")
            print("4. Estadísticas de equipos")
            print("5. Ver tabla de posiciones actual")
            print("6. Volver a elegir liga")
            print("7. Salir")

            choice = input("\nSelecciona una opción (1-7): ").strip()

            if choice == '1':
                self.weekly_predictions_mode()
            elif choice == '2':
                self.jornada_detailed_mode()
            elif choice == '3':
                self.individual_match_mode()
            elif choice == '4':
                self.team_statistics_mode()
            elif choice == '5':
                self.display_current_standings()
            elif choice == '6':
                return 'change_league'
            elif choice == '7':
                print("\n¡Hasta luego!")
                return 'exit'
            else:
                print("ERROR Opción no válida")
                input("Presiona Enter para continuar...")

    # ------------------------------------------------------------ jornadas

    def _pick_matchday(self, season: int, phase: str) -> Optional[int]:
        status = menu_actions.get_jornada_status(
            PROJECT_ROOT, season, data_dir=self.cfg.data_dir_path, phase=phase)
        available = status['available_matchdays']
        if not available:
            print(f"No hay jornadas disponibles para {phase} {season}")
            return None

        print(f"\nFase actual: {phase} {season}")
        print(f"Jornadas disponibles: {min(available)} - {max(available)}")
        print(f"Jornadas completadas: {len(status['finished_jornadas'])}")
        if status['upcoming_jornadas']:
            print(f"Próxima jornada: {min(status['upcoming_jornadas'])}")

        print("\nOpciones:")
        print("1. Seleccionar jornada específica")
        print("2. Siguiente jornada no completada")
        print("3. Ver última jornada completada")
        sub_choice = input("Selecciona opción (1-3): ").strip()

        if sub_choice == '1':
            raw = input("Ingresa número de jornada: ").strip()
            if not raw.isdigit() or int(raw) not in available:
                print("ERROR Jornada no válida")
                return None
            return int(raw)
        if sub_choice == '2':
            matchday = menu_actions.resolve_next_matchday(status)
            if matchday is None:
                print("No hay jornadas pendientes")
            return matchday
        if sub_choice == '3':
            matchday = menu_actions.resolve_last_finished_matchday(status)
            if matchday is None:
                print("No hay jornadas completadas")
            return matchday

        print("Opción no válida")
        return None

    def _predict_matchday(self, matchday: int, season: int, phase: str) -> List[Dict]:
        return predict_week("liga_mx", matchday, season, phase)

    def _display_predictions(self, predictions: List[Dict], matchday: int, phase: str):
        print(f"\nLiga MX - Jornada {matchday} ({phase}) temporada {self._season_phase()[0]}:")
        print("-" * 100)
        for pred in predictions:
            if pred.get('error'):
                print(f"  ERROR: {pred['error']}")
                continue
            home = self._code(pred['home_team'])
            away = self._code(pred['away_team'])
            print(f"  {home} vs {away}: {pred['predicted_result']} "
                  f"({pred['confidence']:.1%}) marcador {pred['most_likely_scoreline']} "
                  f"| goles esp. {pred['expected_goals']:.2f}")

    def weekly_predictions_mode(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("PREDICCIÓN DE JORNADA COMPLETA")
        print("=" * 60)
        season, phase = self._season_phase()
        matchday = self._pick_matchday(season, phase)
        if matchday is None:
            input("Presiona Enter para continuar...")
            return
        try:
            predictions = self._predict_matchday(matchday, season, phase)
            self._display_predictions(predictions, matchday, phase)
        except (RuntimeError, ValueError) as e:
            print(f"ERROR {e}")
        input("\nPresiona Enter para continuar...")

    def jornada_detailed_mode(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("PREDICCIÓN POR JORNADA (DETALLADA)")
        print("=" * 60)
        season, phase = self._season_phase()
        matchday = self._pick_matchday(season, phase)
        if matchday is None:
            input("Presiona Enter para continuar...")
            return

        try:
            predictions = self._predict_matchday(matchday, season, phase)
        except (RuntimeError, ValueError) as e:
            print(f"ERROR {e}")
            input("Presiona Enter para continuar...")
            return

        print(f"\nJornada {matchday} ({phase}) -- detalle por partido:")
        for pred in predictions:
            if pred.get('error'):
                print(f"  ERROR: {pred['error']}")
                continue
            print("\n" + "-" * 70)
            print(f"{pred['home_team']} vs {pred['away_team']}")
            print(f"  Predicción: {pred['predicted_result']} (confianza {pred['confidence']:.1%})")
            for outcome, prob in pred['probabilities'].items():
                print(f"    {outcome}: {prob:.1%}")
            print(f"  Marcador probable: {pred['most_likely_scoreline']} "
                  f"(goles esperados: {pred['expected_goals']:.2f})")
            over_25 = pred['over_under'].get('over_2.5')
            if over_25:
                print(f"  Over/Under 2.5: {over_25['prediction']} ({over_25['over_prob']:.1%} over)")

        print("\n" + "=" * 60)
        send_choice = input("¿Enviar predicciones al dashboard? (s/n): ").strip().lower()
        if send_choice == 's':
            panel_predictions = ensemble_predictions_to_panel_format(
                predictions, self.cfg, self.team_codes, self.team_logos)
            self._panel._send_predictions_to_panel(panel_predictions)

        input("\nPresiona Enter para continuar...")

    # ------------------------------------------------------------- partido

    def _select_team(self, teams: List[str]) -> Optional[str]:
        """Selector de equipo paginado -- copia mínima de
        PredictionMenu._select_team (misma UX), sin instanciar la clase legacy
        completa solo por este helper."""
        print("\nEquipos disponibles:")
        page_size = 10
        current_page = 0
        total_pages = (len(teams) + page_size - 1) // page_size

        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(teams))
            print(f"\nPágina {current_page + 1}/{total_pages}:")
            for i in range(start_idx, end_idx):
                print(f"{i + 1:2d}. {teams[i]}")
            if current_page < total_pages - 1:
                print(f"{end_idx + 1:2d}. Siguiente página →")
            if current_page > 0:
                print(f"{end_idx + 2:2d}. ← Página anterior")

            choice = input(f"\nSelecciona equipo (1-{end_idx + 2}): ").strip()
            if not choice.isdigit():
                print("ERROR Ingresa un número válido")
                continue
            choice_num = int(choice)
            if choice_num == end_idx + 1 and current_page < total_pages - 1:
                current_page += 1
            elif choice_num == end_idx + 2 and current_page > 0:
                current_page -= 1
            elif 1 <= choice_num <= end_idx:
                return teams[choice_num - 1]
            else:
                print("ERROR Opción no válida")

    def individual_match_mode(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("PREDICCIÓN PARTIDO POR PARTIDO")
        print("=" * 60)
        teams = sorted(self.fe.teams['name'].tolist())

        print("\nEquipo LOCAL:")
        home = self._select_team(teams)
        if home is None:
            return
        print("\nEquipo VISITANTE:")
        away = self._select_team(teams)
        if away is None:
            return

        match_date = pd.Timestamp.now().normalize()
        result = predict_match_ensemble(self.outcome_model, self.dc_model, self.glm_model,
                                         self.fe, home, away, match_date)
        if result.get('error'):
            print(f"ERROR {result['error']}")
        else:
            print(f"\n{home} vs {away}")
            print(f"Predicción: {result['predicted_result']} (confianza {result['confidence']:.1%})")
            for outcome, prob in result['probabilities'].items():
                print(f"  {outcome}: {prob:.1%}")
            print(f"Marcador probable: {result['most_likely_scoreline']}")

        input("\nPresiona Enter para continuar...")

    # ------------------------------------------------------ estadísticas

    def team_statistics_mode(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("ESTADÍSTICAS DE EQUIPOS")
        print("=" * 60)
        teams = sorted(self.fe.teams['name'].tolist())
        team = self._select_team(teams)
        if team is None:
            return

        season, phase = self._season_phase()
        stats = menu_actions.get_team_statistics_v2(self.fe, team, season_year=season, phase=phase)

        form = stats['form']
        standings = stats['standings']
        home_perf = stats['home_performance']
        away_perf = stats['away_performance']

        print(f"\nEstadísticas de {team}:")
        print(f"  Posición: {standings['position']}° con {standings['points']} pts "
              f"({standings['played']} jugados)")
        print(f"  Forma (últimos 5): {form['wins']}V {form['draws']}E {form['losses']}D "
              f"(win rate {form['win_rate']:.1%})")
        print(f"  Como local: {home_perf['wins']}V {home_perf['draws']}E {home_perf['losses']}D "
              f"en {home_perf['matches_played']} juegos")
        print(f"  Como visitante: {away_perf['wins']}V {away_perf['draws']}E {away_perf['losses']}D "
              f"en {away_perf['matches_played']} juegos")

        input("\nPresiona Enter para continuar...")

    def display_current_standings(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        season, phase = self._season_phase()
        print(f"TABLA DE POSICIONES -- Liga MX {phase} {season}")
        print("=" * 60)

        table = self.fe.get_standings_table(date=pd.Timestamp.now(), season_year=season, phase=phase)
        if table.empty:
            print("Sin partidos jugados todavía en esta fase.")
        else:
            print(f"{'Pos':<4} {'Equipo':<25} {'PJ':<4} {'Pts':<4} {'DG':<5}")
            for _, row in table.iterrows():
                print(f"{row['position']:<4} {row['team'][:24]:<25} {row['played']:<4} "
                      f"{row['points']:<4} {row['gd']:<5}")

        input("\nPresiona Enter para continuar...")


def main():
    menu = LigaMXMenu()
    if not menu.initialize():
        print("ERROR No se pudo inicializar el sistema de Liga MX")
        return
    menu.display_main_menu()


if __name__ == "__main__":
    main()
