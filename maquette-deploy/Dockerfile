# Static deploy of the Twin RAG WebUI maquette (Sweden bundle).
#
# Pure browser-side prototype: index.html loads React + Babel via CDN
# and all *.jsx files inline. Zero build, just serve the folder.
#
# `operator-overrides.css` is appended to the bundle's styles.css at
# build time so designer-side updates to the Sweden bundle can land
# clean (no merge conflicts on our patches). See operator-overrides.css
# for the diff vs the proto.

FROM caddy:2.8-alpine
COPY site/ /srv/
COPY operator-overrides.css /tmp/operator-overrides.css
RUN cat /tmp/operator-overrides.css >> /srv/styles.css && rm /tmp/operator-overrides.css
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80 443
