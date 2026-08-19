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

# code -> logo URL (media.api-sports.io, league_id=262). Atlante y los
# históricos no aparecieron en las temporadas 2024/2025 de API-Football
# (Atlante estaba en Liga de Expansión hasta el Apertura 2025) -- quedan sin
# logo por ahora; TEAM_LOGOS.get(code, '') en menu_interface.py ya maneja
# el caso de logo faltante igual que hace con Premier.
LIGAMX_TEAM_LOGOS = {
    "AME": "https://media.api-sports.io/football/teams/2287.png",
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
