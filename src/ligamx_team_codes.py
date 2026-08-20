"""Códigos cortos y logos de equipos de Liga MX (equivalente a TEAM_CODES/TEAM_LOGOS
de Premier en menu_interface.py, ver docs/plan_5_ligas_ligamx.md Paso 5).

Liga MX no tiene siglas oficiales estilo FA, así que los códigos se eligieron
a mano para que no colisionen entre sí. Los logos vienen de API-Football
(media.api-sports.io, equivalente al CDN resources.premierleague.com de
Premier) -- ver src/build_liga_mx_dataset.py, league_id=262. Los nombres son
los canónicos de src/ligamx_team_aliases.py (CANONICAL_TEAMS).
"""

LIGAMX_TEAM_CODES = {
    "América": "AME",
    "Atlante": "ATE",
    "Atlas": "ATS",
    "Atlético San Luis": "ASL",
    "Chivas": "CHI",
    "Club Tijuana": "TIJ",
    "Cruz Azul": "CAZ",
    "FC Juárez": "JUA",
    "León": "LEO",
    "Mazatlán FC": "MAZ",
    "Monterrey": "MTY",
    "Necaxa": "NEC",
    "Pachuca": "PAC",
    "Puebla FC": "PUE",
    "Querétaro": "QRO",
    "Santos Laguna": "SAN",
    "Tigres UANL": "TIG",
    "Toluca": "TOL",
    "UNAM Pumas": "PUM",
    # Históricos (descendidos/desaparecidos, quedan en el pool de entrenamiento)
    "Club San Luis": "CSL",
    "Estudiantes Tecos": "TEC",
    "Jaguares de Chiapas": "JAG",
    "Lobos BUAP": "LOB",
    "Monarcas Morelia": "MOR",
    "Veracruz": "VER",
}

# code -> logo URL (media.api-sports.io, league_id=262). Atlante no aparece en
# las temporadas 2024/2025 de API-Football (estaba en Liga de Expansión hasta
# el Apertura 2025) -- ese logo viene de Transfermarkt en vez de API-Football
# (visto en data/ligamx/raw/matches/transfermarkt/html/, perfil del equipo
# https://www.transfermarkt.com/cf-atlante/startseite/verein/6709).
#
# OJO: `TEAM_LOGOS.get(code, '')` (menu_interface.py) sí tolera un logo
# faltante para Premier -- pero safesports-panel *rechaza* el partido entero
# si homeTeamLogo/awayTeamLogo llegan vacíos (zod: `.min(1)` en
# safesports-panel/lib/validations.ts, error "Away team logo is required").
# Un código sin logo acá == esos partidos nunca llegan al dashboard, sin
# aviso más que el conteo de "Errores" al enviar. Verificar que todo código
# de LIGAMX_TEAM_CODES tenga entrada acá al agregar equipos nuevos.
LIGAMX_TEAM_LOGOS = {
    "AME": "https://media.api-sports.io/football/teams/2287.png",
    "ATE": "https://tmssl.akamaized.net/images/wappen/profil/6709.png?lm=1418834720",
    "ATS": "https://media.api-sports.io/football/teams/2283.png",
    "ASL": "https://media.api-sports.io/football/teams/2314.png",
    "CHI": "https://media.api-sports.io/football/teams/2278.png",
    "TIJ": "https://media.api-sports.io/football/teams/2280.png",
    "CAZ": "https://media.api-sports.io/football/teams/2295.png",
    "JUA": "https://media.api-sports.io/football/teams/2298.png",
    "LEO": "https://media.api-sports.io/football/teams/2289.png",
    "MAZ": "https://media.api-sports.io/football/teams/14002.png",
    "MTY": "https://media.api-sports.io/football/teams/2282.png",
    "NEC": "https://media.api-sports.io/football/teams/2288.png",
    "PAC": "https://media.api-sports.io/football/teams/2292.png",
    "PUE": "https://media.api-sports.io/football/teams/2291.png",
    "QRO": "https://media.api-sports.io/football/teams/2290.png",
    "SAN": "https://media.api-sports.io/football/teams/2285.png",
    "TIG": "https://media.api-sports.io/football/teams/2279.png",
    "TOL": "https://media.api-sports.io/football/teams/2281.png",
    "PUM": "https://media.api-sports.io/football/teams/2286.png",
}
