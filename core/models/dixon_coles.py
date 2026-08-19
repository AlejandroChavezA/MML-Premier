"""Dixon-Coles model para marcador de partidos -- liga doméstica (una sola competición).

Port acotado de MML-Mundial/src_v2/models/dixon_coles.py (ver
docs/plan_5_ligas_ligamx.md, Paso 3). La matemática vectorizada de MLE
(NLL + gradiente analítico) se mantiene igual; lo que se quita es todo lo
específico de Mundial:

- COMPETITION_GROUPS / GROUP_KEYS / SAMPLE_WEIGHT_MAP: una liga doméstica es
  una sola competición homogénea, así que gamma (ventaja de local) pasa de
  ser un dict por grupo a un escalar único.
- El prior de ELO para el cold-start de equipos nuevos ya no viene de
  selecciones nacionales (src_v2.data.real_elo / TEAM_TLA, inexistentes en
  este repo) sino de core.elo.EloTracker, ajustado sobre los mismos partidos
  de esta liga.

Referencia: Dixon, M. J., & Coles, S. G. (1997). Modelling association
football scores and inefficiencies in the football betting market.
JRSS Series C, 46(2).
"""
import json
import warnings
from math import exp, log
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from core.elo import EloTracker

warnings.filterwarnings('ignore')

MAX_GOALS = 10
OVER_UNDER_LINES = [0.5, 1.5, 2.5, 3.5]


def most_likely_scoreline(lam_home: float, lam_away: float) -> Tuple[int, int]:
    best = -np.inf
    best_h, best_a = 0, 0
    for h in range(MAX_GOALS + 1):
        ph = exp(stats.poisson.logpmf(h, lam_home)) if lam_home > 0 else (1.0 if h == 0 else 0.0)
        if ph < 1e-10:
            continue
        for a in range(MAX_GOALS + 1):
            pa = exp(stats.poisson.logpmf(a, lam_away)) if lam_away > 0 else (1.0 if a == 0 else 0.0)
            prob = ph * pa
            if prob > best:
                best = prob
                best_h, best_a = h, a
    return best_h, best_a


def _prepare_matches(matches_df: pd.DataFrame, team_index: dict, decay_hl: int) -> tuple:
    """Arrays vectorizados desde el DataFrame de partidos (sin noción de competición/grupo)."""
    home_idx = np.array([team_index[t] for t in matches_df['home_team']])
    away_idx = np.array([team_index[t] for t in matches_df['away_team']])
    hg = np.array(matches_df['home_score'].values, dtype=np.float64)
    ag = np.array(matches_df['away_score'].values, dtype=np.float64)

    now = pd.Timestamp(matches_df['date'].max())
    dates = pd.to_datetime(matches_df['date'].values)
    days_ago = np.array([max(0, (now - d).days) for d in dates], dtype=np.float64)
    weight = np.exp(-days_ago / decay_hl)

    return home_idx, away_idx, hg, ag, weight


def _compute_nll_and_grad(params: np.ndarray, N: int, M: int,
                           home_idx: np.ndarray, away_idx: np.ndarray,
                           hg: np.ndarray, ag: np.ndarray, weight: np.ndarray,
                           reg_lambda: float,
                           alpha_prior: Optional[np.ndarray] = None,
                           beta_prior: Optional[np.ndarray] = None,
                           team_lambda: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """NLL + gradiente analítico vectorizado.

    Layout de parámetros (2N+3 para N equipos):
      0..N-1: alpha (ataque)
      N..2N-1: beta (defensa)
      2N: gamma (ventaja de local, escalar -- una sola competición)
      2N+1: mu (baseline)
      2N+2: rho (correlación marcadores bajos)
    """
    alpha = params[:N]
    beta = params[N:2 * N]
    gamma = params[2 * N]
    mu = params[2 * N + 1]
    rho = params[2 * N + 2]

    alpha_c = alpha - np.mean(alpha)
    beta_c = beta - np.mean(beta)

    lam_h = np.exp(mu + gamma + alpha_c[home_idx] + beta_c[away_idx])
    lam_a = np.exp(mu + alpha_c[away_idx] + beta_c[home_idx])

    lam_h = np.maximum(lam_h, 1e-12)
    lam_a = np.maximum(lam_a, 1e-12)

    s_h = hg / lam_h - 1.0
    s_a = ag / lam_a - 1.0

    tau_factor = np.ones(M)
    dtau_dlh = np.zeros(M)
    dtau_dla = np.zeros(M)
    dtau_drho = np.zeros(M)

    sq = np.sqrt(lam_h * lam_a)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    tau_eps = 1e-6

    valid00 = m00 & ((sq + rho) > tau_eps)
    denom00 = np.maximum(sq[valid00] + rho, tau_eps)
    tau_factor[valid00] = np.maximum(1.0 + rho / np.maximum(sq[valid00], tau_eps), tau_eps)
    dtau_dlh[valid00] = -rho / (2.0 * lam_h[valid00] * denom00)
    dtau_dla[valid00] = -rho / (2.0 * lam_a[valid00] * denom00)
    dtau_drho[valid00] = 1.0 / denom00

    valid01 = m01 & ((lam_h - rho) > tau_eps)
    denom01 = np.maximum(lam_h[valid01] - rho, tau_eps)
    tau_factor[valid01] = np.maximum(1.0 - rho / np.maximum(lam_h[valid01], tau_eps), tau_eps)
    dtau_dlh[valid01] = rho / (lam_h[valid01] * denom01)
    dtau_drho[valid01] = -1.0 / denom01

    valid10 = m10 & ((lam_a - rho) > tau_eps)
    denom10 = np.maximum(lam_a[valid10] - rho, tau_eps)
    tau_factor[valid10] = np.maximum(1.0 - rho / np.maximum(lam_a[valid10], tau_eps), tau_eps)
    dtau_dla[valid10] = rho / (lam_a[valid10] * denom10)
    dtau_drho[valid10] = -1.0 / denom10

    tau_factor[m11] = np.maximum(1.0 + rho, tau_eps)
    dtau_drho[m11] = 1.0 / tau_factor[m11]

    logtau = np.log(tau_factor)

    loglik = logtau + stats.poisson.logpmf(hg, lam_h) + stats.poisson.logpmf(ag, lam_a)
    nll = -np.sum(weight * loglik)

    if alpha_prior is not None and beta_prior is not None:
        alpha_pen = alpha_c - alpha_prior
        beta_pen = beta_c - beta_prior
    else:
        alpha_pen = alpha_c
        beta_pen = beta_c

    if team_lambda is not None:
        lam_alpha = team_lambda
        lam_beta = team_lambda
    else:
        lam_alpha = np.full(N, reg_lambda)
        lam_beta = np.full(N, reg_lambda)

    penalty = float(np.sum(lam_alpha * alpha_pen ** 2) + np.sum(lam_beta * beta_pen ** 2))
    nll += penalty

    grad = np.zeros_like(params)

    dh = s_h + dtau_dlh
    da = s_a + dtau_dla

    contrib_alpha = np.zeros(N)
    np.add.at(contrib_alpha, home_idx, weight * dh * lam_h)
    np.add.at(contrib_alpha, away_idx, weight * da * lam_a)

    contrib_beta = np.zeros(N)
    np.add.at(contrib_beta, home_idx, weight * da * lam_a)
    np.add.at(contrib_beta, away_idx, weight * dh * lam_h)

    grad[:N] = -contrib_alpha + 2 * lam_alpha * alpha_pen
    grad[N:2 * N] = -contrib_beta + 2 * lam_beta * beta_pen

    grad[2 * N] = -np.sum(weight * dh * lam_h)  # gamma (escalar)
    grad[2 * N + 1] = -np.sum(weight * (dh * lam_h + da * lam_a))  # mu
    grad[2 * N + 2] = -np.sum(weight * dtau_drho)  # rho

    return nll, grad


class DixonColesModel:
    """Dixon-Coles con MLE L2-regularizado, para una liga doméstica de club."""

    def __init__(self, decay_hl: int = 180, reg_lambda: float = 0.01, prior_hl: float = 5.0,
                 elo_k_factor: float = 20.0, elo_home_advantage: float = 60.0):
        self.decay_hl = decay_hl
        self.reg_lambda = reg_lambda
        self.prior_hl = prior_hl
        self.elo_k_factor = elo_k_factor
        self.elo_home_advantage = elo_home_advantage

        self.teams: List[str] = []
        self.team_index: Dict[str, int] = {}
        self.alpha: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.gamma: float = 0.0
        self.mu: float = 0.0
        self.rho: float = 0.0

        self._elo_ratings: Dict[str, float] = {}
        self._elo_regression_alpha: Optional[Tuple[float, float]] = None
        self._elo_regression_beta: Optional[Tuple[float, float]] = None
        self._gpg_regression_alpha: Optional[Tuple[float, float]] = None
        self._alpha_prior: Optional[np.ndarray] = None
        self._beta_prior: Optional[np.ndarray] = None

        self._team_lambda: Optional[np.ndarray] = None
        self._team_avg_gpg: Optional[np.ndarray] = None
        self._temperature: float = 1.0
        self._home_idx: Optional[np.ndarray] = None
        self._away_idx: Optional[np.ndarray] = None
        self._hg: Optional[np.ndarray] = None
        self._ag: Optional[np.ndarray] = None
        self._weight: Optional[np.ndarray] = None

    def _nll_with_grad(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        N = len(self.teams)
        M = len(self._home_idx)
        team_lambda = self._team_lambda
        return _compute_nll_and_grad(
            params, N, M, self._home_idx, self._away_idx, self._hg, self._ag, self._weight,
            self.reg_lambda, alpha_prior=self._alpha_prior, beta_prior=self._beta_prior,
            team_lambda=team_lambda,
        )

    def fit(self, matches_df: pd.DataFrame, team_names: Optional[List[str]] = None) -> dict:
        """Ajuste en dos etapas: MLE sin regularizar -> prior GPG/ELO -> reoptimizar."""
        all_teams = sorted(set(matches_df['home_team'].tolist() + matches_df['away_team'].tolist()))
        self.teams = sorted(set(team_names or all_teams) | set(all_teams))
        self.team_index = {t: i for i, t in enumerate(self.teams)}
        N = len(self.teams)

        self._home_idx, self._away_idx, self._hg, self._ag, self._weight = _prepare_matches(
            matches_df, self.team_index, self.decay_hl)
        M = len(self._home_idx)

        home_counts = np.bincount(self._home_idx, minlength=N)
        away_counts = np.bincount(self._away_idx, minlength=N)
        match_counts = home_counts + away_counts
        K = self.prior_hl
        self._team_lambda = self.reg_lambda * (K / (match_counts.astype(np.float64) + K))
        self._match_counts = match_counts

        avg_total = matches_df['home_score'].mean() + matches_df['away_score'].mean()
        mu_init = log(max(avg_total / 2.0, 0.1))

        x0 = np.zeros(2 * N + 3)
        x0[2 * N + 1] = mu_init
        bounds = [(None, None)] * (2 * N + 2) + [(-0.999, 0.999)]

        self._alpha_prior = None
        self._beta_prior = None
        old_tl = self._team_lambda
        self._team_lambda = None
        saved_reg = self.reg_lambda
        self.reg_lambda = 1e-6

        result1 = minimize(self._nll_with_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                            options={'maxiter': 20000, 'ftol': 1e-12})

        self.reg_lambda = saved_reg
        self._team_lambda = old_tl

        self._unpack(result1.x, N)

        # Etapa 2: prior GPG (alpha) + ELO (beta), reoptimizar
        self._fit_gpg_regression(matches_df)
        self._fit_elo_regression(matches_df)
        self._compute_gpg_priors()
        self._compute_elo_priors()

        if self._alpha_prior is not None and self._beta_prior is not None:
            x0_2 = np.concatenate([self._alpha_prior, self._beta_prior,
                                    [result1.x[2 * N], self.mu, self.rho]])
            result2 = minimize(self._nll_with_grad, x0_2, method='L-BFGS-B', jac=True, bounds=bounds,
                                options={'maxiter': 20000, 'ftol': 1e-12})
            result = result2
        else:
            result = result1

        self._unpack(result.x, N)

        return {
            'success': bool(result.success),
            'n_iter': int(result.nit),
            'final_nll': float(result.fun),
            'n_teams': N,
            'n_matches': M,
            'message': str(result.message),
        }

    def _unpack(self, x: np.ndarray, N: int):
        alpha_raw = x[:N]
        beta_raw = x[N:2 * N]
        self.alpha = alpha_raw - np.mean(alpha_raw)
        self.beta = beta_raw - np.mean(beta_raw)
        self.gamma = float(x[2 * N])
        self.mu = float(x[2 * N + 1])
        self.rho = float(x[2 * N + 2])

    # -- Priors (cold start para equipos nuevos / poco historial) --------

    def _fit_elo_regression(self, matches_df: pd.DataFrame, min_matches: int = 10):
        """Aprende ELO -> alpha y ELO -> beta a partir de EloTracker de esta misma liga."""
        elo = EloTracker(k_factor=self.elo_k_factor, home_advantage=self.elo_home_advantage).fit(matches_df)
        self._elo_ratings = elo.snapshot()

        X, y_alpha, y_beta = [], [], []
        for team in self.teams:
            rating = self._elo_ratings.get(team)
            if rating is None:
                continue
            i = self.team_index[team]
            X.append([rating])
            y_alpha.append(self.alpha[i])
            y_beta.append(self.beta[i])

        if len(X) < 20:
            self._elo_regression_alpha = (0.0, 0.0)
            self._elo_regression_beta = (0.0, 0.0)
            return

        X_arr = np.array(X)
        reg_a = LinearRegression().fit(X_arr, y_alpha)
        reg_b = LinearRegression().fit(X_arr, y_beta)
        self._elo_regression_alpha = (float(reg_a.intercept_), float(reg_a.coef_[0]))
        self._elo_regression_beta = (float(reg_b.intercept_), float(reg_b.coef_[0]))

    def _fit_gpg_regression(self, matches_df: pd.DataFrame, min_matches: int = 10):
        """Aprende goles-por-partido-promedio -> alpha (sin acoplamiento a Mundial)."""
        home_gf = matches_df.groupby('home_team')['home_score'].sum()
        away_gf = matches_df.groupby('away_team')['away_score'].sum()
        goals_for = home_gf.add(away_gf, fill_value=0)

        home_mp = matches_df.groupby('home_team').size()
        away_mp = matches_df.groupby('away_team').size()
        matches_played = home_mp.add(away_mp, fill_value=0)

        avg_gpg = goals_for / matches_played

        N = len(self.teams)
        self._team_avg_gpg = np.zeros(N, dtype=np.float64)
        for team in self.teams:
            self._team_avg_gpg[self.team_index[team]] = avg_gpg.get(team, 0.0)

        X, y = [], []
        for team in self.teams:
            if matches_played.get(team, 0) < min_matches:
                continue
            i = self.team_index[team]
            X.append([avg_gpg.get(team, 0.0)])
            y.append(self.alpha[i])

        if len(X) < 20:
            self._gpg_regression_alpha = (0.0, 1.0)
            return

        reg = LinearRegression().fit(np.array(X), y)
        self._gpg_regression_alpha = (float(reg.intercept_), float(reg.coef_[0]))

    def _compute_elo_priors(self):
        if self._elo_regression_beta is None:
            self._beta_prior = None
            return
        N = len(self.teams)
        beta_p = np.zeros(N, dtype=np.float64)
        inter_b, slope_b = self._elo_regression_beta
        for team in self.teams:
            rating = self._elo_ratings.get(team, self.elo_home_advantage and 1500.0)
            beta_p[self.team_index[team]] = inter_b + slope_b * rating
        self._beta_prior = beta_p - np.mean(beta_p)

    def _compute_gpg_priors(self):
        if self._gpg_regression_alpha is None or self._team_avg_gpg is None:
            self._alpha_prior = None
            return
        inter, slope = self._gpg_regression_alpha
        alpha_p = inter + slope * self._team_avg_gpg
        self._alpha_prior = alpha_p - np.mean(alpha_p)

    def cold_start_params(self, team: str) -> Tuple[float, float]:
        """Estima alpha, beta para un equipo fuera del training set, vía ELO."""
        if self._elo_regression_alpha is None:
            return (0.0, 0.0)
        rating = self._elo_ratings.get(team, 1500.0)
        inter_a, slope_a = self._elo_regression_alpha
        inter_b, slope_b = self._elo_regression_beta
        return (inter_a + slope_a * rating, inter_b + slope_b * rating)

    # -- Predicción --------------------------------------------------

    def predict(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Devuelve lambda_home, lambda_away para un partido."""
        h_idx = self.team_index.get(home_team)
        a_idx = self.team_index.get(away_team)

        if h_idx is None or a_idx is None:
            alpha_h, beta_h = self.cold_start_params(home_team) if h_idx is None else (
                self.alpha[h_idx], self.beta[h_idx])
            alpha_a, beta_a = self.cold_start_params(away_team) if a_idx is None else (
                self.alpha[a_idx], self.beta[a_idx])
            lam_h = exp(self.mu + self.gamma + alpha_h + beta_a)
            lam_a = exp(self.mu + alpha_a + beta_h)
            return lam_h, lam_a

        lam_h = exp(self.mu + self.gamma + self.alpha[h_idx] + self.beta[a_idx])
        lam_a = exp(self.mu + self.alpha[a_idx] + self.beta[h_idx])
        return lam_h, lam_a

    def scoreline_probs(self, lam_home: float, lam_away: float, max_goals: int = MAX_GOALS,
                         n: int = 8) -> List[dict]:
        scores = []
        for h in range(max_goals + 1):
            ph = stats.poisson.pmf(h, lam_home) if lam_home > 0 else (1.0 if h == 0 else 0.0)
            if ph < 1e-8:
                continue
            for a in range(max_goals + 1):
                pa = stats.poisson.pmf(a, lam_away) if lam_away > 0 else (1.0 if a == 0 else 0.0)
                prob = ph * pa
                if prob >= 0.005:
                    scores.append({'score': f'{h}-{a}', 'prob': round(float(prob), 4),
                                    'home_goals': h, 'away_goals': a})
        scores.sort(key=lambda x: x['prob'], reverse=True)
        return scores[:n]

    def outcome_probs(self, lam_home: float, lam_away: float) -> Dict[str, float]:
        p_home, p_draw, p_away = 0.0, 0.0, 0.0
        for h in range(MAX_GOALS + 1):
            ph = stats.poisson.pmf(h, lam_home) if lam_home > 0 else (1.0 if h == 0 else 0.0)
            if ph < 1e-10:
                continue
            for a in range(MAX_GOALS + 1):
                pa = stats.poisson.pmf(a, lam_away) if lam_away > 0 else (1.0 if a == 0 else 0.0)
                prob = ph * pa
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob
        total = p_home + p_draw + p_away
        raw = {
            'home': p_home / total if total > 0 else 1 / 3,
            'draw': p_draw / total if total > 0 else 1 / 3,
            'away': p_away / total if total > 0 else 1 / 3,
        }
        if self._temperature != 1.0:
            raw = self._apply_temp(raw)
        return {k: round(v, 4) for k, v in raw.items()}

    def over_under(self, lam_total: float) -> Dict[str, dict]:
        ou = {}
        for k in OVER_UNDER_LINES:
            prob = 1.0 - stats.poisson.cdf(int(k), lam_total)
            ou[f'over_{k}'] = {'prob': round(float(prob), 4), 'prediction': 'OVER' if prob >= 0.5 else 'UNDER'}
        return ou

    def full_predict(self, home_team: str, away_team: str) -> dict:
        lam_h, lam_a = self.predict(home_team, away_team)
        ml_h, ml_a = most_likely_scoreline(lam_h, lam_a)
        lam_t = lam_h + lam_a
        return {
            'expectedGoals': {'home': round(lam_h, 2), 'away': round(lam_a, 2), 'total': round(lam_t, 2)},
            'mostLikelyScoreline': f'{ml_h}-{ml_a}',
            'topScorelines': self.scoreline_probs(lam_h, lam_a),
            'poissonOutcome': self.outcome_probs(lam_h, lam_a),
            'overUnder': self.over_under(lam_t),
        }

    def _apply_temp(self, probs: Dict[str, float]) -> Dict[str, float]:
        eps = 1e-10
        z = np.array([np.log(max(probs['home'], eps)), np.log(max(probs['draw'], eps)),
                      np.log(max(probs['away'], eps))])
        z_s = z / self._temperature
        z_s = z_s - np.max(z_s)
        p_cal = np.exp(z_s) / np.sum(np.exp(z_s))
        return {'home': round(float(p_cal[0]), 4), 'draw': round(float(p_cal[1]), 4), 'away': round(float(p_cal[2]), 4)}

    # -- Serialización -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'teams': self.teams,
            'alpha': {t: float(self.alpha[i]) if self.alpha is not None else 0.0 for i, t in enumerate(self.teams)},
            'beta': {t: float(self.beta[i]) if self.beta is not None else 0.0 for i, t in enumerate(self.teams)},
            'gamma': self.gamma,
            'mu': self.mu,
            'rho': self.rho,
            'decay_hl': self.decay_hl,
            'reg_lambda': self.reg_lambda,
            'prior_hl': self.prior_hl,
            'elo_k_factor': self.elo_k_factor,
            'elo_home_advantage': self.elo_home_advantage,
            'elo_ratings': self._elo_ratings,
            'elo_regression_alpha': list(self._elo_regression_alpha) if self._elo_regression_alpha else None,
            'elo_regression_beta': list(self._elo_regression_beta) if self._elo_regression_beta else None,
            'gpg_regression_alpha': list(self._gpg_regression_alpha) if self._gpg_regression_alpha else None,
            'temperature': self._temperature,
        }

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        model = cls(decay_hl=data.get('decay_hl', 180), reg_lambda=data.get('reg_lambda', 0.01),
                    prior_hl=data.get('prior_hl', 5.0), elo_k_factor=data.get('elo_k_factor', 20.0),
                    elo_home_advantage=data.get('elo_home_advantage', 60.0))
        model.teams = data['teams']
        model.team_index = {t: i for i, t in enumerate(model.teams)}
        model.alpha = np.array([data['alpha'][t] for t in model.teams])
        model.beta = np.array([data['beta'][t] for t in model.teams])
        model.gamma = data['gamma']
        model.mu = data['mu']
        model.rho = data['rho']
        model._elo_ratings = data.get('elo_ratings', {})
        if data.get('elo_regression_alpha'):
            model._elo_regression_alpha = tuple(data['elo_regression_alpha'])
            model._elo_regression_beta = tuple(data['elo_regression_beta'])
        if data.get('gpg_regression_alpha'):
            model._gpg_regression_alpha = tuple(data['gpg_regression_alpha'])
        model._temperature = data.get('temperature', 1.0)
        return model
