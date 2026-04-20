# API de Predicciones

## Endpoints

### Importar Predicciones
```
POST https://safesports-panel.vercel.app/api/predictions/import
```

### Autenticación
Usar API Key en el header:
```
Authorization: Bearer sk_tu_api_key_aqui
```

### Formato del Body
```json
{
  "predictions": [
    {
      "homeTeam": "Arsenal FC",
      "awayTeam": "Liverpool FC",
      "matchDate": "2026-04-25T15:00:00Z",
      "prediction": "LOCAL",
      "predictionCode": 2,
      "confidence": 0.56,
      "odds": 1.85,
      "sport": "football",
      "league": "Premier League",
      "round": 34,
      "status": "SCHEDULED",
      "markets": {
        "over_2.5": {
          "prediction": "OVER",
          "overProb": 0.81,
          "underProb": 0.19,
          "odds": 1.90
        }
      }
    }
  ]
}
```

### Códigos de Predicción
- `0` = VISITANTE
- `1` = EMPATE  
- `2` = LOCAL

### Estados
- `SCHEDULED` = Partido no jugado
- `FINISHED` = Partido terminado
- `CANCELLED` = Partido cancelado

### Tipos de Liga para soccer
- `Premier League`
- `Champions League`
- `Liga MX`

### Deportes soportados (sport)
- `nfl`
- `nba`
- `mlb`
- `nhl`
- `soccer`

---

## Ejemplo con curl

```bash
curl -X POST "https://safesports-panel.vercel.app/api/predictions/import" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk_tu_api_key" \
  -d '{
    "predictions": [
      {
        "homeTeam": "Arsenal FC",
        "awayTeam": "Liverpool FC",
        "matchDate": "2026-04-25T15:00:00Z",
        "prediction": "LOCAL",
        "predictionCode": 2,
        "confidence": 0.56,
        "odds": 1.85,
        "sport": "football",
        "league": "Premier League",
        "round": 34,
        "status": "SCHEDULED"
      }
    ]
  }'
```

### Respuesta Exitosa
```json
{
  "success": true,
  "message": "Imported 1 predictions",
  "imported": 1,
  "total": 1
}
```

---

## Obtener API Key

### Opción 1: Desde el Panel
1. Login en https://safesports-panel.vercel.app
2. Ir a Perfil → Settings
3. Generate API Key

### Opción 2: Programáticamente
```bash
# 1. Login para obtener token
curl -X POST "https://safesports-panel.vercel.app/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"nickname":"tu_usuario","password":"tu_password"}'

# 2. Generar API key
curl -X POST "https://safesports-panel.vercel.app/api/auth/api-key/generate" \
  -H "Content-Type: application/json" \
  -d '{"email":"tu@email.com","password":"tu_password"}'
```

---

## Campos del Prediction

| Campo | Tipo | Requerido | Descripción |
|-------|------|----------|-------------|
| homeTeam | string | ✅ | Nombre del equipo local |
| awayTeam | string | ✅ | Nombre del equipo visitante |
| matchDate | string | ✅ | Fecha del partido (ISO 8601) |
| prediction | string | ✅ | LOCAL, EMPATE, o VISITANTE |
| predictionCode | number | ✅ | 0, 1, o 2 |
| confidence | number | ✅ | Confianza (0-1) |
| odds | number | ❌ | Cuota decimal |
| sport | string | ✅ | football, basketball, etc. |
| league | string | ✅ | Nombre de laliga |
| round | number | ❌ | Jornada/Round |
| status | string | ✅ | SCHEDULED, FINISHED, CANCELLED |
| homeScore | number | ❌ | Goles del equipo local |
| awayScore | number | ❌ | Goles del equipo visitante |
| markets | object | ❌ | Mercados de apuestas |