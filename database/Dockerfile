FROM postgres:17

COPY ./01-schema.sql /docker-entrypoint-initdb.d/
COPY ./02-dummy_data.sql /docker-entrypoint-initdb.d/
