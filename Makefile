# Makefile
.PHONY: up down build logs ps prune

build:
\tdocker compose build

up:
\tdocker compose up -d

down:
\tdocker compose down

logs:
\tdocker compose logs -f --tail=200

ps:
\tdocker compose ps

prune:
\tdocker compose down -v --remove-orphans
\tdocker system prune -f