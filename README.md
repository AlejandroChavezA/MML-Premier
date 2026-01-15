# Premier League Data Analysis Project

Proyecto completo para el análisis de datos de la Premier League utilizando Python y Machine Learning.

## 🚀 Configuración Rápida

### 1. Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv premier-league-env

# Activar (Mac/Linux)
source premier-league-env/bin/activate

# Activar (Windows)
premier-league-env\Scripts\activate
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Obtener API Key
Regístrate en [Football-Data.org](https://www.football-data.org/login) para obtener tu API key gratuita.

## 📁 Estructura del Proyecto

```
premier-league/
├── premier-league-env/          # Entorno virtual
├── data/                         # Datos recolectados
│   ├── teams.csv                # Información de equipos
│   ├── matches_2023.csv         # Partidos temporada 2023
│   └── standings_2023.csv       # Tabla de posiciones
├── src/                         # Scripts principales
│   ├── data_collection.py       # Recolección de datos
│   ├── analysis.py              # Análisis estadístico
│   └── visualization.py          # Visualizaciones
├── notebooks/                   # Jupyter notebooks
│   └── analysis.ipynb          # Análisis interactivo
├── requirements.txt              # Dependencias
└── README.md                    # Este archivo
```

## 📊 Uso del Proyecto

### 1. Recolección de Datos
```python
from src.data_collection import PremierLeagueDataCollector

# Crear colector
collector = PremierLeagueDataCollector()

# Configurar tu API key
collector.headers['X-Auth-Token'] = 'TU_API_KEY_AQUI'

# Obtener datos
collector.get_premier_league_teams()
collector.get_premier_league_matches(2023)
collector.get_standings(2023)
```

### 2. Análisis de Datos
```python
from src.analysis import PremierLeagueAnalyzer

# Crear analizador
analyzer = PremierLeagueAnalyzer()

# Cargar datos
analyzer.load_data(2023)

# Estadísticas básicas
stats = analyzer.basic_statistics()
print(stats)

# Análisis de rendimiento
home_stats, away_stats = analyzer.team_performance_analysis()
```

### 3. Visualizaciones
```python
from src.visualization import PremierLeagueVisualizer

# Crear visualizador
visualizer = PremierLeagueVisualizer()

# Cargar datos
visualizer.load_data(2023)

# Crear gráficos
visualizer.plot_standings_table()
visualizer.plot_goals_scatter()
visualizer.plot_goal_trends()
visualizer.plot_home_away_performance()
```

## 🎯 Características Principales

### Recolección de Datos
- ✅ Equipos de la Premier League
- ✅ Partidos por temporada
- ✅ Tabla de posiciones
- ✅ Estadísticas detalladas

### Análisis Estadístico
- ✅ Estadísticas básicas de temporada
- ✅ Análisis de rendimiento por equipo
- ✅ Distribución de goles
- ✅ Comparación local vs visitante

### Visualizaciones Interactivas
- ✅ Tabla de posiciones interactiva
- ✅ Gráficos de dispersión de goles
- ✅ Tendencias durante la temporada
- ✅ Comparación rendimiento local/visitante
- ✅ Gráficos de pastel de resultados

## 🛠️ Librerías Utilizadas

- **pandas**: Manejo y análisis de datos
- **matplotlib**: Visualizaciones básicas
- **plotly**: Gráficos interactivos
- **requests**: Peticiones HTTP a APIs
- **beautifulsoup4**: Web scraping
- **seaborn**: Visualizaciones estadísticas
- **numpy**: Cálculos numéricos
- **scikit-learn**: Machine Learning
- **streamlit**: Aplicaciones web

## 📈 Próximos Pasos

### Análisis Avanzado
- [ ] Modelos de predicción de resultados
- [ ] Análisis de jugadores individuales
- [ ] Estadísticas avanzadas (xG, xA)
- [ ] Clustering de equipos

### Machine Learning
- [ ] Modelo de clasificación de resultados
- [ ] Sistema de recomendación de partidos
- [ ] Análisis de sentimiento de noticias
- [ ] Predicción de lesiones

### Visualizaciones
- [ ] Dashboard interactivo con Streamlit
- [ ] Mapas de calor de estadios
- [ ] Animaciones de goles
- [ ] Gráficos 3D de estadísticas

## 🤝 Contribuir

1. Fork del proyecto
2. Crear rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📝 Notas

- La API de Football-Data.org tiene un límite de 10 peticiones por minuto en el plan gratuito
- Los datos se guardan automáticamente en la carpeta `/data`
- Los gráficos se guardan como archivos PNG y también se muestran interactivamente

## 🐛 Problemas Comunes

**Error: "No se encontraron archivos de datos"**
- Asegúrate de haber ejecutado primero el script de recolección de datos
- Verifica que tu API key sea válida

**Error: "Import pandas could not be resolved"**
- Activa el entorno virtual antes de ejecutar los scripts
- Reinstala las dependencias con `pip install -r requirements.txt`

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.