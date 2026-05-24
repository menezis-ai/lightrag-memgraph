# Static deploy of the Twin RAG WebUI maquette.
#
# Pure browser-side prototype: index.html loads React + Babel via CDN
# and all *.jsx files inline. Zero build, just serve the folder.
#
# `source/` holds the full SPA (HTML + JSX + CSS + data.js + db.jsx).
# Everything is versioned in this repo — no external dependency on a
# designer-side bundle. `operator-overrides.css` is appended to
# source/styles.css at image build time.

FROM caddy:2.8-alpine
COPY source/ /srv/
COPY operator-overrides.css /tmp/operator-overrides.css
RUN cat /tmp/operator-overrides.css >> /srv/styles.css && rm /tmp/operator-overrides.css
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80 443
