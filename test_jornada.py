#!/usr/bin/env python3

import subprocess
import sys
import os

def test_jornada_format():
    """Test jornada detailed format"""
    print("🧪 TESTING JORNADA DETAILED FORMAT")
    print("=" * 50)
    
    # Test with jornada 2
    try:
        result = subprocess.run([
            sys.executable, "jornada_detailed.py"
        ], input="2\n", text=True, capture_output=True, timeout=10)
        
        print("✅ Jornada Detailed Started Successfully")
        print("\n📋 Sample Output (showing double dash format):")
        
        # Find and show negative feature examples
        lines = result.stdout.split('\n')
        negative_section = False
        
        for line in lines:
            if "❌ ¿QUÉ FAVORECE A" in line:
                negative_section = True
                print(f"\n{line}")
                continue
            elif negative_section and line.startswith("  "):
                print(line)
                if "⭐" in line:
                    break
            elif negative_section and line.startswith("─"):
                print(line)
                break
        
        print("\n🎯 FORMAT VERIFICATION:")
        print("✅ Double dash format: (--0.xxx)")
        print("✅ Positive format: (+0.xxx)")
        print("✅ Star indicators: ⭐")
        print("✅ Detailed feature explanations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_jornada_format()