# Static deploy of the Twin RAG WebUI maquette (Sweden bundle).
#
# Pure browser-side prototype: index.html loads React + Babel via CDN
# and all *.jsx files inline. Zero build, just serve the folder.

FROM caddy:2.8-alpine
COPY site/ /srv/
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80 443
