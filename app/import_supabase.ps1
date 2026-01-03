# ==============================
# CONFIGURACIÓN
# ==============================

$PG_BIN = "C:\Program Files\PostgreSQL\18\bin"
$BACKUP_FILE = "backup.sql"

$SUPABASE_URL = "postgresql://postgres@db.qmfneumbqunkhtqmursm.supabase.co:5432/postgres?sslmode=require"

# ==============================
# EJECUCIÓN
# ==============================

Write-Host "📦 Importando backup a Supabase..." -ForegroundColor Cyan

Set-Location $PG_BIN

psql $SUPABASE_URL -f $BACKUP_FILE

Write-Host "✅ Importación finalizada" -ForegroundColor Green
