# Langfuse en local (Docker Compose)

Cada alumno levanta su propio Langfuse en su portátil. `docker-compose.yml` es una copia **sin cambios** del
fichero oficial (https://github.com/langfuse/langfuse/blob/main/docker-compose.yml, commit indicado en la primera
línea); toda la configuración va en `.env`. Guía oficial: https://langfuse.com/self-hosting/deployment/docker-compose

Probado el 2026-10-14 con Langfuse 4.58.0, Docker 28.4, Docker Compose 2.39 y el SDK de Python `langfuse` 4.17.

## Requisitos

- Docker con Compose v2. En Windows, Docker Desktop con WSL2; instálalo antes de la clase.
- Unos 2,2 GB de RAM con Langfuse en marcha (medido: web 770 MB, ClickHouse 625 MB, worker 590 MB, MinIO 120 MB,
  Postgres 70 MB y Redis 12 MB) y unos 5,5 GB de disco para las imágenes. No usa la GPU.
- Puertos libres: 3000 (interfaz y API) y 9090; y, solo en `127.0.0.1`, 3030, 5432, 6379, 8123, 9000 y 9091.

## Levantarlo

Desde la raíz del proyecto, con el `.env` ya creado (`cp .env.example .env`):

```bash
cd servidor/langfuse
./generate_env.sh               # crea servidor/langfuse/.env y escribe LANGFUSE_HOST y las claves en el .env del proyecto
docker compose up -d
curl http://localhost:3000/api/public/health   # {"status":"OK",...} tras 1-3 minutos
```

`generate_env.sh` genera secretos aleatorios (contraseñas de Postgres, ClickHouse, Redis y MinIO, `NEXTAUTH_SECRET`,
`SALT`, `ENCRYPTION_KEY`) y usa la **inicialización automática** (`LANGFUSE_INIT_*`) para que, al arrancar, se creen
la organización, el proyecto `RAG semana 2`, sus claves y un usuario administrador. Al terminar, muestra el usuario y
la contraseña para entrar en http://localhost:3000.

En el `.env` del proyecto pon también tu nombre en `ALUMNO`: se guarda como *User ID* en cada traza.

Para pararlo: `docker compose down`. Los datos quedan en volúmenes de Docker; `docker compose down -v` los borra.
Si se borra `servidor/langfuse/.env`, hay que borrar también los volúmenes, porque las contraseñas cambian.

## Sin Docker

No pasa nada: sin `LANGFUSE_HOST` y las claves en `.env`, `rag_lab/tracing.py` guarda las trazas en
`trazas/trazas.jsonl` y todos los notebooks funcionan igual, también `casa/casa_01_observabilidad`.

## Alternativa: una instancia compartida en el servidor

Los mismos pasos sirven en el servidor de clase cambiando la URL: `./generate_env.sh http://IP_DEL_SERVIDOR:3000`,
y abriendo solo los puertos 3000 y 9090. Limitación: la edición libre no permite crear un proyecto y unas claves por
alumno por API (eso son la Org API y la *Instance Management API*, solo de la Enterprise Edition:
https://langfuse.com/docs/administration/scim-and-org-api,
https://langfuse.com/self-hosting/administration/instance-management-api). Habría que usar un proyecto compartido:
`tracing.py` ya pone en cada traza `user_id = ALUMNO`, la etiqueta `alumno:<ALUMNO>` y la versión del RAG, así que
cada alumno filtra por *User ID* en la interfaz o con `langfuse.api.observations.get_many(user_id=..., version=...)`.

## Notas de la versión 4

- El SDK de Python v4 lee `LANGFUSE_BASE_URL`; `LANGFUSE_HOST` sigue funcionando pero está marcada como obsoleta.
  `tracing.py` acepta las dos.
- Los endpoints antiguos `/api/public/traces` no están disponibles en el modo de datos de la v4; las trazas se
  consultan con `/api/public/observations` (`langfuse.api.observations.get_many`).
