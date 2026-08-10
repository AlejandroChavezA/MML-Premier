"""Mapeo canonico de nombres de equipo para Liga MX / Liga de Expansion MX.

Cada fuente de datos nombra a los equipos distinto:
- API-Football (build_liga_mx_dataset.py, 2016-2024_liga_mx.csv)
- FBref via mirror de GitHub (build_ligamx_fbref_dataset.py)
- Transfermarkt (build_ligamx_transfermarkt_dataset.py)
- openfootball/world (historial de Ascenso MX / Liga de Expansion, plan_5_ligas_ligamx.md)

Este modulo centraliza esas variantes en un solo nombre canonico por equipo,
para poder cruzar/deduplicar partidos entre fuentes. Si aparece un nombre no
registrado, `normalize_team_name` falla ruidosamente en vez de dejarlo pasar
sin normalizar (evita que un alias nuevo se cuele silenciosamente como
"equipo distinto" en el dataset consolidado).
"""

from __future__ import annotations

# Equipos activos en Liga MX (primera division) a la fecha de este mapeo.
# Atlante volvio a primera para el Apertura 2025 tras jugar en Liga de
# Expansion MX (segunda division); ver ALIASES_SEGUNDA_DIVISION mas abajo
# para su historial previo.
CANONICAL_LIGA_MX = [
    "América",
    "Atlante",
    "Atlas",
    "Atlético San Luis",
    "Chivas",
    "Club Tijuana",
    "Cruz Azul",
    "FC Juárez",
    "León",
    "Mazatlán FC",
    "Monterrey",
    "Necaxa",
    "Pachuca",
    "Puebla FC",
    "Querétaro",
    "Santos Laguna",
    "Tigres UANL",
    "Toluca",
    "UNAM Pumas",
]

# Equipos que jugaron Liga MX en 2016-2021 y luego descendieron o
# desaparecieron. Se mantienen para no perder historial en el pool de
# entrenamiento de esas temporadas.
CANONICAL_HISTORICOS = [
    "Club San Luis",
    "Estudiantes Tecos",
    "Jaguares de Chiapas",
    "Lobos BUAP",
    "Monarcas Morelia",
    "Veracruz",
]

CANONICAL_TEAMS = CANONICAL_LIGA_MX + CANONICAL_HISTORICOS

# variante exacta (como aparece en la fuente) -> nombre canonico.
# Incluye el propio nombre canonico como variante identidad.
TEAM_ALIASES: dict[str, str] = {
    # --- América ---
    "América": "América",
    "America": "América",
    "Club America": "América",
    "CF América": "América",
    # --- Atlante ---
    "Atlante": "Atlante",
    "CF Atlante": "Atlante",
    # --- Atlas ---
    "Atlas": "Atlas",
    "Atlas Guadalajara": "Atlas",
    # --- Atlético San Luis ---
    "Atlético San Luis": "Atlético San Luis",
    "Atletico San Luis": "Atlético San Luis",
    "Atlético de San Luis": "Atlético San Luis",
    "San Luis": "Atlético San Luis",
    "Atletico": "Atlético San Luis",
    "Atlético": "Atlético San Luis",
    # --- Chivas / Guadalajara ---
    "Chivas": "Chivas",
    "Guadalajara": "Chivas",
    "Guadalajara Chivas": "Chivas",
    "Deportivo Guadalajara": "Chivas",
    # --- Club Tijuana ---
    "Club Tijuana": "Club Tijuana",
    "Tijuana": "Club Tijuana",
    # --- Cruz Azul ---
    "Cruz Azul": "Cruz Azul",
    "Club Cruz Azul": "Cruz Azul",
    "CD Cruz Azul": "Cruz Azul",
    # --- FC Juárez ---
    "FC Juárez": "FC Juárez",
    "FC Juarez": "FC Juárez",
    # --- León ---
    "León": "León",
    "Leon": "León",
    "Club León FC": "León",
    "Club León": "León",
    # --- Mazatlán FC ---
    "Mazatlán FC": "Mazatlán FC",
    "Mazatlán": "Mazatlán FC",
    "Mazatlan": "Mazatlán FC",
    # --- Monterrey ---
    "Monterrey": "Monterrey",
    "CF Monterrey": "Monterrey",
    # --- Necaxa ---
    "Necaxa": "Necaxa",
    "Club Necaxa": "Necaxa",
    # --- Pachuca ---
    "Pachuca": "Pachuca",
    "CF Pachuca": "Pachuca",
    # --- Puebla FC ---
    "Puebla FC": "Puebla FC",
    "Puebla": "Puebla FC",
    "Club Puebla": "Puebla FC",
    # --- Querétaro ---
    "Querétaro": "Querétaro",
    "Queretaro": "Querétaro",
    "Club Queretaro": "Querétaro",
    "Querétaro FC": "Querétaro",
    "Gallos Blancos": "Querétaro",
    # --- Santos Laguna ---
    "Santos Laguna": "Santos Laguna",
    "Santos": "Santos Laguna",
    # --- Tigres UANL ---
    "Tigres UANL": "Tigres UANL",
    "UANL": "Tigres UANL",
    "UANL Tigres": "Tigres UANL",
    # --- Toluca ---
    "Toluca": "Toluca",
    "Deportivo Toluca": "Toluca",
    # --- UNAM Pumas ---
    "UNAM Pumas": "UNAM Pumas",
    "UNAM": "UNAM Pumas",
    "Pumas UNAM": "UNAM Pumas",
    "U.N.A.M. - Pumas": "UNAM Pumas",
    # --- Historicos (descendidos/desaparecidos) ---
    "Jaguares de Chiapas": "Jaguares de Chiapas",
    "Jaguares Chiapas": "Jaguares de Chiapas",
    "Chiapas FC": "Jaguares de Chiapas",  # mismo club, distinto nombre comercial por temporada
    "Lobos BUAP": "Lobos BUAP",
    "Lobos Buap": "Lobos BUAP",
    "Monarcas Morelia": "Monarcas Morelia",
    "Monarcas": "Monarcas Morelia",
    "Veracruz": "Veracruz",
    "CD Veracruz": "Veracruz",
    "Club San Luis": "Club San Luis",  # club distinto de Atlético San Luis (desaparecio ~2013)
    "Estudiantes Tecos": "Estudiantes Tecos",
}

# Equipos de Liga de Expansion MX / Ascenso MX (segunda division), extraidos
# de los 30 archivos openfootball 2010-11 a 2024-25 (mx2ascenso hasta
# 2019-20, mx2expansion desde 2020-21). Atlante/Club Tijuana/FC Juarez/etc.
# ya estan en TEAM_ALIASES arriba porque tambien jugaron en primera; aqui
# solo van los equipos que SOLO se vieron en segunda division.
ALIASES_SEGUNDA_DIVISION: dict[str, str] = {
    "Venados FC": "Venados FC",
    "Tepatitlán": "Tepatitlán",
    "Tapatío": "Tapatío",
    "Coyotes de Tlaxcala": "Coyotes de Tlaxcala",
    "Atlético La Paz": "Atlético La Paz",
    "Mineros de Zacatecas": "Mineros de Zacatecas",
    "Dorados de Sinaloa": "Dorados de Sinaloa",
    "Tampico Madero": "Tampico Madero",
    "Atlético Morelia": "Atlético Morelia",
    "Club Celaya": "Club Celaya",
    "Correcaminos UAT": "Correcaminos UAT",
    "Cancún FC": "Cancún FC",
    "Alebrijes de Oaxaca": "Alebrijes de Oaxaca",
    "Alacranes de Durango": "Alacranes de Durango",
    "Albinegros Orizaba": "Albinegros Orizaba",
    "Atlante UTN": "Atlante",  # Atlante con nombre de patrocinador en 2010-11
    "Atlético Zacatepec": "Zacatepec",
    "Club Zacatepec": "Zacatepec",  # mismo club que Atlético Zacatepec, distintas temporadas
    "Ballenas Galeana": "Ballenas Galeana",
    "CF Monterrey II": "CF Monterrey II",
    "Cafetaleros de Chiapas": "Cafetaleros de Chiapas",
    "Cafetaleros de Tapachula": "Cafetaleros de Tapachula",
    "Cimarrones de Sonora": "Cimarrones de Sonora",
    "Club Irapuato": "Club Irapuato",
    "Cruz Azul Hidalgo": "Cruz Azul Hidalgo",
    "Delfines del Carmen": "Delfines del Carmen",
    "Deportivo Tepic": "Deportivo Tepic",
    "Estudiantes de Altamira": "Estudiantes de Altamira",
    "Guerreros Hermosillo": "Guerreros Hermosillo",
    "Indios Juárez": "Indios Juárez",
    "Leones Negros": "Leones Negros",
    "Loros UCOL": "Loros UCOL",
    "Murciélagos FC": "Murciélagos FC",
    "Mérida FC": "Mérida FC",
    "Neza FC": "Neza FC",
    "Potros UAEM": "Potros UAEM",
    "Pumas Morelos": "Pumas Morelos",
    "Pumas Tabasco": "Pumas Tabasco",
    "Reboceros de la Piedad": "Reboceros de la Piedad",
    "TM Fútbol Club": "TM Fútbol Club",
}


def normalize_team_name(name: str, *, source: str = "") -> str:
    """Devuelve el nombre canonico de un equipo dado cualquier variante conocida.

    Lanza ValueError si el nombre no esta registrado, en vez de devolverlo
    tal cual, para que un alias nuevo (equipo recien ascendido, cambio de
    nombre de patrocinio, etc.) se detecte al construir el dataset y no se
    cuele como un "equipo" distinto silenciosamente.
    """
    key = name.strip()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    if key in ALIASES_SEGUNDA_DIVISION:
        return ALIASES_SEGUNDA_DIVISION[key]
    where = f" (fuente: {source})" if source else ""
    raise ValueError(
        f"Nombre de equipo no registrado en TEAM_ALIASES{where}: {name!r}. "
        "Agregalo a src/ligamx_team_aliases.py antes de seguir."
    )
