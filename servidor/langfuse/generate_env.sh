#!/usr/bin/env bash
# Crea .env con secretos aleatorios para el docker-compose.yml oficial de Langfuse y, si existe el .env del
# proyecto (dos carpetas más arriba), escribe en él LANGFUSE_HOST y las claves.
# Uso: ./generate_env.sh [URL] [EMAIL_ADMIN]     (por defecto http://localhost:3000 y admin@example.com)
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then echo ".env ya existe; bórralo si quieres regenerarlo (y borra los datos con 'docker compose down -v')."; exit 1; fi

URL="${1:-http://localhost:3000}"
EMAIL="${2:-admin@example.com}"
hex() { openssl rand -hex "$1"; }

PG=$(hex 16); MINIO=$(hex 16)
PK="pk-lf-$(hex 16)"; SK="sk-lf-$(hex 16)"; PASS=$(hex 12)
cat > .env <<EOT
NEXTAUTH_URL=$URL
TELEMETRY_ENABLED=false
NEXTAUTH_SECRET=$(hex 32)
SALT=$(hex 16)
ENCRYPTION_KEY=$(hex 32)
POSTGRES_PASSWORD=$PG
DATABASE_URL=postgresql://postgres:$PG@postgres:5432/postgres
CLICKHOUSE_PASSWORD=$(hex 16)
REDIS_AUTH=$(hex 16)
MINIO_ROOT_PASSWORD=$MINIO
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MINIO
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MINIO
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MINIO
LANGFUSE_INIT_ORG_ID=iab4
LANGFUSE_INIT_ORG_NAME=IAB4
LANGFUSE_INIT_PROJECT_ID=rag-semana2
LANGFUSE_INIT_PROJECT_NAME=RAG semana 2
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=$PK
LANGFUSE_INIT_PROJECT_SECRET_KEY=$SK
LANGFUSE_INIT_USER_EMAIL=$EMAIL
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=$PASS
EOT
chmod 600 .env
echo "Creado servidor/langfuse/.env."

PROJECT_ENV=../../.env
if [ -f "$PROJECT_ENV" ]; then
  for name in LANGFUSE_HOST LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
    grep -q "^$name=" "$PROJECT_ENV" || echo "$name=" >> "$PROJECT_ENV"
  done
  sed -i "s#^LANGFUSE_HOST=.*#LANGFUSE_HOST=$URL#; s#^LANGFUSE_PUBLIC_KEY=.*#LANGFUSE_PUBLIC_KEY=$PK#; s#^LANGFUSE_SECRET_KEY=.*#LANGFUSE_SECRET_KEY=$SK#" "$PROJECT_ENV"
  echo "Escritas LANGFUSE_HOST y las claves en el .env del proyecto."
else
  echo "No hay .env en el proyecto (cp .env.example .env). Añade estas líneas cuando lo crees:"
  echo "LANGFUSE_HOST=$URL"; echo "LANGFUSE_PUBLIC_KEY=$PK"; echo "LANGFUSE_SECRET_KEY=$SK"
fi
echo "Interfaz: $URL  |  usuario: $EMAIL  |  contraseña: $PASS"
