# Static deploy of the Twin RAG WebUI maquette (Sweden bundle).
#
# Pure browser-side prototype: index.html loads React + Babel via CDN
# and all *.jsx files inline. Zero build, just serve the folder.
#
# `operator-overrides.css` is appended to the bundle's styles.css at
# build time. `patches/site-overlay/` carries individual file overlays
# (documents.jsx, activity.jsx, data.js extended for doc validation +
# audit feed) that ship on top of the designer's Sweden bundle.
# Keeping these patches outside the source bundle means the designer
# can update Downloads/ freely without losing operator-side additions.

FROM caddy:2.8-alpine
COPY site/ /srv/
COPY patches/site-overlay/ /srv/
COPY operator-overrides.css /tmp/operator-overrides.css
RUN cat /tmp/operator-overrides.css >> /srv/styles.css && rm /tmp/operator-overrides.css
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80 443
