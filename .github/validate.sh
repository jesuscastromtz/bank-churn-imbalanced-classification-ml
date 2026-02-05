#!/usr/bin/env bash

# 🤖 Script de Validación para Agentes
# Ejecutar después de que cada agente complete su tarea

set -e

echo "🔍 Validando estructura Bank Churn Project..."
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASS++))
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((FAIL++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# =============================================================================
# 1. VALIDACIONES DE ESTRUCTURA
# =============================================================================

echo -e "${YELLOW}=== VALIDANDO ESTRUCTURA ===${NC}"
echo ""

# Notebooks
echo "Notebooks:"
if [ -f "notebooks/1_problema_analisis.ipynb" ]; then
    check_pass "1_problema_analisis.ipynb existe"
else
    check_fail "1_problema_analisis.ipynb NO EXISTE"
fi

if [ -f "notebooks/2_solucion_modelo.ipynb" ]; then
    check_pass "2_solucion_modelo.ipynb existe"
else
    check_fail "2_solucion_modelo.ipynb NO EXISTE"
fi

if [ -f "notebooks/3_resultados.ipynb" ]; then
    check_pass "3_resultados.ipynb existe"
else
    check_fail "3_resultados.ipynb NO EXISTE"
fi

if [ ! -f "notebooks/notebook.ipynb" ]; then
    check_pass "notebook.ipynb (viejo) eliminado"
else
    check_fail "notebook.ipynb (viejo) DEBE ser eliminado"
fi

# src/ files
echo ""
echo "Archivos en src/:"
if [ -f "src/core.py" ]; then
    check_pass "src/core.py existe"
else
    check_fail "src/core.py NO EXISTE"
fi

if [ -f "src/__init__.py" ]; then
    check_pass "src/__init__.py existe"
else
    check_fail "src/__init__.py NO EXISTE"
fi

# Verificar que NO existan archivos bloat
if [ ! -f "src/churn_graphics.py" ]; then
    check_pass "src/churn_graphics.py (bloat) eliminado"
else
    check_fail "src/churn_graphics.py (bloat) DEBE ser eliminado"
fi

if [ ! -f "src/visualization.py" ]; then
    check_pass "src/visualization.py (bloat) eliminado"
else
    check_fail "src/visualization.py (bloat) DEBE ser eliminado"
fi

if [ ! -f "src/data_understanding.py" ]; then
    check_pass "src/data_understanding.py (bloat) eliminado"
else
    check_fail "src/data_understanding.py (bloat) DEBE ser eliminado"
fi

if [ ! -f "src/data_preparation.py" ]; then
    check_pass "src/data_preparation.py (bloat) eliminado"
else
    check_fail "src/data_preparation.py (bloat) DEBE ser eliminado"
fi

# Contar archivos en src/
echo ""
echo "Conteo de archivos en src/:"
SRC_COUNT=$(ls src/*.py 2>/dev/null | wc -l)
if [ "$SRC_COUNT" -eq 2 ]; then
    check_pass "src/ tiene exactamente 2 archivos .py"
else
    check_fail "src/ debe tener 2 archivos .py (tiene $SRC_COUNT)"
fi

# Directorios
echo ""
echo "Directorios requeridos:"
if [ -d "data/sample" ]; then
    check_pass "data/sample/ existe"
else
    check_fail "data/sample/ NO EXISTE"
fi

if [ -d "visualizations" ]; then
    check_pass "visualizations/ existe"
else
    check_fail "visualizations/ NO EXISTE"
fi

# =============================================================================
# 2. VALIDACIONES DE CÓDIGO
# =============================================================================

echo ""
echo -e "${YELLOW}=== VALIDANDO CÓDIGO ===${NC}"
echo ""

# Verificar imports en notebooks
echo "Imports en notebooks:"
if grep -q "from src import" notebooks/1_problema_analisis.ipynb; then
    check_pass "1_problema_analisis.ipynb importa correctamente"
else
    check_warn "1_problema_analisis.ipynb no usa 'from src import'"
fi

if grep -q "from src import" notebooks/2_solucion_modelo.ipynb; then
    check_pass "2_solucion_modelo.ipynb importa correctamente"
else
    check_warn "2_solucion_modelo.ipynb no usa 'from src import'"
fi

if grep -q "from src import" notebooks/3_resultados.ipynb; then
    check_pass "3_resultados.ipynb importa correctamente"
else
    check_warn "3_resultados.ipynb no usa 'from src import'"
fi

# Verificar funciones en core.py
echo ""
echo "Funciones en src/core.py:"
if grep -q "def error_count" src/core.py; then
    check_pass "error_count() existe"
else
    check_fail "error_count() NO EXISTE"
fi

if grep -q "def evaluate_model" src/core.py; then
    check_pass "evaluate_model() existe"
else
    check_fail "evaluate_model() NO EXISTE"
fi

if grep -q "def upsample" src/core.py; then
    check_pass "upsample() existe"
else
    check_fail "upsample() NO EXISTE"
fi

if grep -q "def downsample" src/core.py; then
    check_pass "downsample() existe"
else
    check_fail "downsample() NO EXISTE"
fi

# Verificar exports en __init__.py
echo ""
echo "Exports en src/__init__.py:"
if grep -q "error_count" src/__init__.py; then
    check_pass "error_count está exportado"
else
    check_fail "error_count NO ESTÁ EXPORTADO"
fi

if grep -q "evaluate_model" src/__init__.py; then
    check_pass "evaluate_model está exportado"
else
    check_fail "evaluate_model NO ESTÁ EXPORTADO"
fi

if grep -q "upsample" src/__init__.py; then
    check_pass "upsample está exportado"
else
    check_fail "upsample NO ESTÁ EXPORTADO"
fi

if grep -q "downsample" src/__init__.py; then
    check_pass "downsample está exportado"
else
    check_fail "downsample NO ESTÁ EXPORTADO"
fi

# =============================================================================
# 3. VALIDACIONES DE TESTS
# =============================================================================

echo ""
echo -e "${YELLOW}=== VALIDANDO TESTS ===${NC}"
echo ""

if [ -f "test_basic.py" ]; then
    check_pass "test_basic.py existe"
    
    # Intentar ejecutar tests
    if python test_basic.py 2>&1 | grep -q "PASSED\|OK"; then
        check_pass "Tests PASAN"
    else
        check_warn "Tests no ejecutados o con advertencias (revisar)"
    fi
else
    check_fail "test_basic.py NO EXISTE"
fi

# =============================================================================
# 4. VALIDACIONES DE DEPENDENCIAS
# =============================================================================

echo ""
echo -e "${YELLOW}=== VALIDANDO DEPENDENCIAS ===${NC}"
echo ""

if [ -f "requirements.txt" ]; then
    check_pass "requirements.txt existe"
    
    # Contar líneas en requirements.txt
    REQ_COUNT=$(wc -l < requirements.txt)
    if [ "$REQ_COUNT" -le 10 ]; then
        check_pass "requirements.txt tiene ≤10 dependencias ($REQ_COUNT)"
    else
        check_warn "requirements.txt tiene $REQ_COUNT líneas (recomendación: ≤10)"
    fi
else
    check_fail "requirements.txt NO EXISTE"
fi

# =============================================================================
# 5. VALIDACIONES DE DOCUMENTACIÓN
# =============================================================================

echo ""
echo -e "${YELLOW}=== VALIDANDO DOCUMENTACIÓN ===${NC}"
echo ""

if [ -f "README.md" ]; then
    check_pass "README.md existe"
    
    # Contar líneas del README
    README_LINES=$(wc -l < README.md)
    if [ "$README_LINES" -lt 100 ]; then
        check_pass "README.md es conciso ($README_LINES líneas)"
    else
        check_warn "README.md es largo ($README_LINES líneas, recomendación: <100 para minimalista)"
    fi
    
    # Verificar keywords
    if grep -qi "problem\|solution\|result\|churn" README.md; then
        check_pass "README.md contiene palabras clave de narrativa"
    else
        check_warn "README.md podría mejorar su narrativa"
    fi
else
    check_fail "README.md NO EXISTE"
fi

# =============================================================================
# 6. RESUMEN
# =============================================================================

echo ""
echo -e "${YELLOW}=== RESUMEN ===${NC}"
echo ""
echo -e "${GREEN}Pasadas: $PASS${NC}"
echo -e "${RED}Fallidas: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ VALIDACIÓN COMPLETA - PROYECTO LISTO${NC}"
    echo ""
    echo "La estructura es MINIMALISTA y NARRATIVA."
    echo "Ready to merge! 🚀"
    exit 0
else
    echo -e "${RED}❌ VALIDACIÓN FALLIDA - REVISAR ARRIBA${NC}"
    echo ""
    echo "Por favor revisar las validaciones marcadas con ❌"
    exit 1
fi
