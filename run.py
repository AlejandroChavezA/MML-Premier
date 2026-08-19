#!/usr/bin/env python3
"""CLI equivalente al menú interactivo (python main.py).

Expone las mismas capacidades que PredictionMenu llamando a las mismas
funciones de src/menu_actions.py, para que menú y CLI nunca diverjan en
comportamiento (ver docs/plan_5_ligas_ligamx.md, Paso 2).
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import menu_actions
from menu_interface import PredictionMenu
from season_utils import get_current_season


def _init_menu() -> PredictionMenu:
    menu = PredictionMenu()
    if not menu.initialize():
        print("ERROR No se pudo inicializar el sistema. Intenta: python main.py --train")
        sys.exit(1)
    return menu


def cmd_jornada(args):
    # Premier sigue exactamente el camino legacy de siempre -- cero cambios
    # de comportamiento. Cualquier otra liga (ej. liga_mx) usa el ensamble
    # multi-liga (core.predict_week), que es el único camino que existe para
    # ligas sin modelo legacy propio.
    if args.league != "premier_league":
        return _cmd_jornada_multileague(args)

    menu = _init_menu()
    season = get_current_season()
    status = menu_actions.get_jornada_status(menu.project_root, season)

    if args.matchday is not None:
        matchday = args.matchday
        if matchday not in status['available_matchdays']:
            print(f"ERROR Jornada {matchday} no válida")
            sys.exit(1)
    elif args.next:
        matchday = menu_actions.resolve_next_matchday(status)
        if matchday is None:
            print("No hay jornadas pendientes")
            sys.exit(1)
    elif args.last_finished:
        matchday = menu_actions.resolve_last_finished_matchday(status)
        if matchday is None:
            print("No hay jornadas completadas")
            sys.exit(1)
    else:
        print("ERROR Especifica --matchday N, --next o --last-finished")
        sys.exit(1)

    predictions = menu_actions.predict_jornada(menu.predictor, matchday, season, menu.current_model)
    if predictions and 'error' in predictions[0]:
        print(f"ERROR Error: {predictions[0]['error']}")
        sys.exit(1)

    summary = menu_actions.summarize_jornada(predictions, menu.current_model)
    print(f"Jornada {matchday} ({menu.current_model}):")
    for pred in predictions:
        if 'error' in pred:
            continue
        home = menu.get_team_code(pred['home_team'])
        away = menu.get_team_code(pred['away_team'])
        print(f"  {home} vs {away}: {pred['predicted_result']} ({pred['confidence']:.1%})")
    print(f"\nResumen: local={summary.get('local_wins')} "
          f"visita={summary.get('away_wins')} empate={summary.get('draws')} "
          f"confianza_prom={summary.get('avg_confidence', 0):.1%}")

    if args.send:
        menu.current_matchday = matchday
        menu.send_to_dashboard(predictions, matchday)


def _cmd_jornada_multileague(args):
    """Camino nuevo (Paso 4/5): cualquier liga config-driven vía core.predict_week."""
    sys.path.insert(0, str(project_root))
    from core.league_config import load_league_config
    from core.panel_export import ensemble_predictions_to_panel_format
    from core.predict_week import predict_week
    from menu_interface import TEAM_METADATA
    from season_utils import get_current_phase

    try:
        cfg = load_league_config(args.league)
    except FileNotFoundError:
        print(f"ERROR Liga desconocida: {args.league}")
        sys.exit(1)

    season = args.season if args.season is not None else get_current_season(league_cfg=cfg)
    phase = args.phase if args.phase else get_current_phase(cfg)
    if cfg.season_type == "split" and phase is None:
        print(f"ERROR '{args.league}' requiere --phase AP|CL (no se pudo auto-resolver)")
        sys.exit(1)

    status = menu_actions.get_jornada_status(project_root, season, data_dir=cfg.data_dir_path, phase=phase)

    if args.matchday is not None:
        matchday = args.matchday
        if matchday not in status['available_matchdays']:
            print(f"ERROR Jornada {matchday} no válida")
            sys.exit(1)
    elif args.next:
        matchday = menu_actions.resolve_next_matchday(status)
        if matchday is None:
            print("No hay jornadas pendientes")
            sys.exit(1)
    elif args.last_finished:
        matchday = menu_actions.resolve_last_finished_matchday(status)
        if matchday is None:
            print("No hay jornadas completadas")
            sys.exit(1)
    else:
        print("ERROR Especifica --matchday N, --next o --last-finished")
        sys.exit(1)

    try:
        predictions = predict_week(args.league, matchday, season, phase)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR {e}")
        sys.exit(1)

    from core.history import append_predictions
    append_predictions(cfg.slug, matchday, season, predictions)

    team_codes, team_logos = TEAM_METADATA.get(cfg.slug, ({}, {}))
    header = f"{cfg.name} - Jornada {matchday}"
    if phase:
        header += f" ({phase})"
    header += f" temporada {season}:"
    print(header)
    for pred in predictions:
        home = team_codes.get(pred['home_team'], pred['home_team'][:3].upper())
        away = team_codes.get(pred['away_team'], pred['away_team'][:3].upper())
        print(f"  {home} vs {away}: {pred['predicted_result']} ({pred['confidence']:.1%}) "
              f"marcador {pred['most_likely_scoreline']}")

    if args.send:
        panel_predictions = ensemble_predictions_to_panel_format(predictions, cfg, team_codes, team_logos)
        menu = PredictionMenu()  # sin .initialize(): _send_predictions_to_panel no usa modelos
        menu._send_predictions_to_panel(panel_predictions)


def cmd_match(args):
    menu = _init_menu()
    match_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now().normalize()
    prediction = menu_actions.predict_single_match(menu.predictor, args.home, args.away, match_date, menu.current_model)
    if 'error' in prediction:
        print(f"ERROR Error: {prediction['error']}")
        sys.exit(1)

    print(f"{args.home} vs {args.away} ({match_date.strftime('%Y-%m-%d')})")
    print(f"Predicción: {prediction['predicted_result']} (confianza {prediction['confidence']:.1%})")
    for outcome, prob in prediction['probabilities'].items():
        print(f"  {outcome}: {prob:.1%}")


def cmd_standings(args):
    menu = _init_menu()
    season = get_current_season()
    standings = menu_actions.load_standings(menu.project_root, season)
    print(f"{'Pos':<4} {'Equipo':<25} {'PJ':<3} {'PTS':<4}")
    for _, row in standings.iterrows():
        print(f"{row['position']:<4} {row['team'][:24]:<25} {row['played_games']:<3} {row['points']:<4}")


def cmd_stats(args):
    menu = _init_menu()
    stats = menu_actions.get_team_statistics(menu.feature_engineer, args.team)
    form = stats['form']
    standings = stats['standings']
    print(f"Estadísticas de {args.team}:")
    print(f"  Forma: {form['wins']}V {form['draws']}E {form['losses']}D (win rate {form['win_rate']:.1%})")
    print(f"  Posición: {standings['position']}° con {standings['points']} pts")


def cmd_models(args):
    menu = _init_menu()
    if args.set:
        available = [m['name'] for m in menu_actions.list_models_with_accuracy(menu.predictor, menu.current_model)]
        if args.set not in available:
            print(f"ERROR Modelo no válido. Disponibles: {', '.join(available)}")
            sys.exit(1)
        menu.current_model = args.set
    models = menu_actions.list_models_with_accuracy(menu.predictor, menu.current_model)
    for m in models:
        marker = " (ACTUAL)" if m['is_current'] else ""
        print(f"  {m['name']}{marker} - accuracy {m['test_accuracy']:.3f}")


def cmd_performance(args):
    menu = _init_menu()
    report = menu_actions.get_model_performance_report(menu.predictor, menu._get_historical_accuracy())
    if not report:
        print("ERROR No hay datos de rendimiento disponibles")
        sys.exit(1)
    for model_name, metrics in report['performance'].items():
        print(f"{model_name}: test={metrics['test_accuracy']:.3f} cv={metrics['cv_mean']:.3f}")
    print(f"\nMejor modelo: {report['best_model']}")


def cmd_export_history(args):
    menu = _init_menu()
    panel_predictions = menu.export_history_to_panel_format()
    if not panel_predictions:
        print("No hay historial disponible")
        return

    export_file = menu.project_root / "predictions_for_panel.json"
    import json
    with open(export_file, 'w') as f:
        json.dump(panel_predictions, f, indent=2)
    print(f"Exportado a: {export_file} ({len(panel_predictions)} predicciones)")

    if args.send:
        menu._send_predictions_to_panel(panel_predictions)


def main():
    parser = argparse.ArgumentParser(description="CLI del predictor Premier League (equivalente al menú interactivo)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_jornada = sub.add_parser("jornada", help="Predicción de una jornada completa")
    group = p_jornada.add_mutually_exclusive_group(required=True)
    group.add_argument("--matchday", type=int, help="Número de jornada")
    group.add_argument("--next", action="store_true", help="Siguiente jornada no completada")
    group.add_argument("--last-finished", action="store_true", help="Última jornada completada")
    p_jornada.add_argument("--send", action="store_true", help="Enviar predicciones al dashboard")
    p_jornada.add_argument("--league", default="premier_league",
                            help="Liga (config/leagues/*.yaml). Default: premier_league")
    p_jornada.add_argument("--season", type=int, help="Override de temporada (default: auto)")
    p_jornada.add_argument("--phase", choices=["AP", "CL"], help="Fase (solo ligas split, ej. liga_mx). Default: auto")
    p_jornada.set_defaults(func=cmd_jornada)

    p_match = sub.add_parser("match", help="Predicción partido por partido")
    p_match.add_argument("--home", required=True)
    p_match.add_argument("--away", required=True)
    p_match.add_argument("--date", help="YYYY-MM-DD (default: hoy)")
    p_match.set_defaults(func=cmd_match)

    p_standings = sub.add_parser("standings", help="Tabla de posiciones actual")
    p_standings.set_defaults(func=cmd_standings)

    p_stats = sub.add_parser("stats", help="Estadísticas de un equipo")
    p_stats.add_argument("--team", required=True)
    p_stats.set_defaults(func=cmd_stats)

    p_models = sub.add_parser("models", help="Listar/cambiar modelo de predicción")
    p_models.add_argument("--set", help="Nombre del modelo a usar")
    p_models.set_defaults(func=cmd_models)

    p_perf = sub.add_parser("performance", help="Rendimiento de modelos")
    p_perf.set_defaults(func=cmd_performance)

    p_export = sub.add_parser("export-history", help="Exportar historial de predicciones al panel")
    p_export.add_argument("--send", action="store_true", help="Enviar al panel tras exportar")
    p_export.set_defaults(func=cmd_export_history)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
