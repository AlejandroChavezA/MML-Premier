#!/usr/bin/env python3
"""
Google Scraper para predicciones de fútbol
Usa Playwright para obtener predicciones del widget de Google.

Usage:
    python scraper.py --home "Arsenal" --away "Chelsea"
    python scraper.py --match "Arsenal vs Chelsea"
"""

import asyncio
import argparse
import sys
from typing import Optional, Dict, List
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Error: Playwright no está instalado")
    print("Ejecuta: pip install playwright && playwright install chromium")
    sys.exit(1)


class GooglePredictionScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.timeout = 30000  # 30 segundos
        
    async def scrape_prediction(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Obtiene predicciones de Google para un partido.
        
        Returns:
            Dict con:
            - home_prediction: float (porcentaje)
            - draw_prediction: float (porcentaje)  
            - away_prediction: float (porcentaje)
            - predicted_winner: str
            - source: str
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                channel='chromium'
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US'
            )
            page = await context.new_page()
            
            try:
                # Construir query de búsqueda
                query = f"{home_team} vs {away_team} premier league prediction"
                search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&hl=en"
                
                print(f"Buscando: {query}")
                
                await page.goto(search_url, timeout=self.timeout)
                
                # Esperar a que cargue la página
                await page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(2)  # Esperar que carguen scripts
                
                # Verificar si hay CAPTCHA
                if await self._check_for_captcha(page):
                    print("⚠️ Google requiere CAPTCHA. Intenta más tarde o usa VPN.")
                    await browser.close()
                    return None
                
                # Esperar a que cargue el widget de predicciones
                prediction_data = await self._extract_prediction(page, home_team, away_team)
                
                await browser.close()
                return prediction_data
                
            except Exception as e:
                print(f"Error durante scraping: {e}")
                await browser.close()
                return None
    
    async def _check_for_captcha(self, page) -> bool:
        """Verifica si hay CAPTCHA"""
        captcha_selectors = [
            '#captcha',
            '[id*="captcha"]',
            '.captcha',
            'text=CAPTCHA',
            'text=我不是机器人',
            'text=I am not a robot'
        ]
        for selector in captcha_selectors:
            try:
                if await page.query_selector(selector):
                    return True
            except:
                pass
        return False
    
    async def _extract_prediction(self, page, home_team: str, away_team: str) -> Optional[Dict]:
        """Extrae la predicción del widget de Google"""
        
        # Selectores conocidos del widget de predicciones de Google
        # ⚠️ Estos selectores cambian frecuentemente - verificar periódicamente
        
        selectors = [
            # Selector para el widget de football predictions
            '[data-ved*="prediction"]',
            '.imso-hu__scr-team',
            '[class*="imso-hu"]',
            # Alternativos
            '.football-card',
            '[data-soccer-prediction]',
            '.sp-c-fixture',
        ]
        
        try:
            # Primero verificar si hay widget
            page_content = await page.content()
            print(f"DEBUG: Página cargada, contenido length: {len(page_content)}")
            
            # Intentar múltiples selectores
            for selector in selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=3000)
                    if element:
                        print(f"DEBUG: Encontré selector: {selector}")
                        return await self._parse_widget(page, home_team, away_team)
                except Exception as e:
                    print(f"DEBUG: Selector {selector} falló: {e}")
                    continue
            
            # Si no encontramos widget específico, buscar en la página
            return await self._parse_page_content(page, home_team, away_team)
            
        except Exception as e:
            print(f"No se pudo extraer predicción: {e}")
            return None
    
    async def _parse_widget(self, page, home_team: str, away_team: str) -> Dict:
        """Parse el widget de predicciones已知"""
        
        # Obtener nombres de equipos del widget
        home_elem = await page.query_selector('[class*="imso-hu__tm-nm"]')
        away_elem = await page.query_selector('[class*="imso-hu__tm-nm"]:nth-child(2)')
        
        # Obtener probabilidades
        percentages = await page.query_selector_all('[class*="imso-hu__percentage"]')
        
        predictions = {}
        
        if len(percentages) >= 3:
            home_pct = await percentages[0].inner_text()
            draw_pct = await percentages[1].inner_text()
            away_pct = await percentages[2].inner_text()
            
            predictions = {
                'home_prediction': self._parse_percentage(home_pct),
                'draw_prediction': self._parse_percentage(draw_pct),
                'away_prediction': self._parse_percentage(away_pct),
                'predicted_winner': self._determine_winner(
                    home_team, away_team, predictions
                ),
                'source': 'google_widget'
            }
        
        return predictions
    
    async def _parse_page_content(self, page, home_team: str, away_team: str) -> Optional[Dict]:
        """Busca predicciones en el contenido de la página"""
        
        content = await page.content()
        
        # Buscar patrones conocidos de probabilidades
        import re
        
        # Patrón para probabilidades tipo "45% - 30% - 25%"
        pattern = r'(\d+)%\s*[-–]\s*(\d+)%\s*[-–]\s*(\d+)%'
        matches = re.findall(pattern, content)
        
        if matches:
            home, draw, away = matches[0]
            predictions = {
                'home_prediction': int(home),
                'draw_prediction': int(draw),
                'away_prediction': int(away),
                'source': 'google_content'
            }
            predictions['predicted_winner'] = self._determine_winner(
                home_team, away_team, predictions
            )
            return predictions
        
        return None
    
    def _parse_percentage(self, text: str) -> int:
        """Convierte texto de porcentaje a entero"""
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0
    
    def _determine_winner(self, home_team: str, away_team: str, predictions: Dict) -> str:
        """Determina el ganador basado en probabilidades"""
        home = predictions.get('home_prediction', 0)
        away = predictions.get('away_prediction', 0)
        draw = predictions.get('draw_prediction', 0)
        
        max_pct = max(home, away, draw)
        
        if max_pct == home:
            return home_team
        elif max_pct == away:
            return away_team
        else:
            return 'DRAW'
    
    async def scrape_multiple(self, matches: List[tuple]) -> List[Dict]:
        """Obtiene predicciones para múltiples partidos"""
        results = []
        
        for home, away in matches:
            print(f"\n{'='*50}")
            result = await self.scrape_prediction(home, away)
            if result:
                results.append({
                    'home': home,
                    'away': away,
                    **result
                })
            await asyncio.sleep(2)  # Esperar entre requests
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Scraper de predicciones de Google')
    parser.add_argument('--home', help='Equipo local')
    parser.add_argument('--away', help='Equipo visitante')
    parser.add_argument('--match', help='Partido en formato "Equipo vs Equipo"')
    parser.add_argument('--headless', action='store_true', default=True, help='Ejecutar headless')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Mostrar navegador')
    
    args = parser.parse_args()
    
    if args.match:
        parts = args.match.split(' vs ')
        if len(parts) != 2:
            print("Error: Formato inválido. Usa 'Equipo vs Equipo'")
            sys.exit(1)
        home, away = parts[0].strip(), parts[1].strip()
    elif args.home and args.away:
        home, away = args.home, args.away
    else:
        parser.print_help()
        sys.exit(1)
    
    scraper = GooglePredictionScraper(headless=args.headless)
    result = asyncio.run(scraper.scrape_prediction(home, away))
    
    if result:
        print(f"\n{'='*50}")
        print(f"PREDICCIONES PARA: {home} vs {away}")
        print(f"{'='*50}")
        print(f"  {home}: {result['home_prediction']}%")
        print(f"  Empate:  {result['draw_prediction']}%")
        print(f"  {away}: {result['away_prediction']}%")
        print(f"\n  Predicción: {result['predicted_winner']}")
        print(f"  Fuente: {result['source']}")
    else:
        print("No se pudo obtener la predicción")


if __name__ == "__main__":
    main()
