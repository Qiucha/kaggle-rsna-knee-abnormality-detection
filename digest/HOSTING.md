# Hosting the training digest

Run these commands from the repository root.

## Navigable research site

```bash
.venv/bin/python scripts/build_digest_site.py          # rebuild the site (idempotent)
.venv/bin/python scripts/build_digest_site.py export   # refresh digest/site/data/ from checkpoints/, then rebuild
python3 -m http.server 8765 --bind 127.0.0.1 --directory .
```

Open <http://127.0.0.1:8765/digest/> (the new `digest/index.html` redirects to `digest/site/index.html`). The site also works straight from the file system: open `digest/index.html`. Serving the repository root keeps the rendered documents' links to configs and scripts working. Every file in `digest/` must have a row in `scripts/digest_site/catalog.py`, or the build fails and lists the missing files.

## Foreground server

```bash
source .venv/bin/activate
python -m http.server 8765 --bind 127.0.0.1 --directory digest
```

Open:

- This machine: <http://127.0.0.1:8765/translated-report-training.html>

Press `Ctrl+C` in the server terminal to stop it.

## Background server

```bash
nohup .venv/bin/python -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory digest \
  > /tmp/novo-digest-http.log 2>&1 &
echo $! > /tmp/novo-digest-http.pid
```

Verify it:

```bash
curl -I http://127.0.0.1:8765/translated-report-training.html
```

Stop that background server:

```bash
kill "$(cat /tmp/novo-digest-http.pid)"
```

The report is self-contained: its MRI preview, styles, and charts are embedded in the HTML, so serving the single file does not require a build step or external assets.

## Exposure note

The localhost binding intentionally prevents access from other machines. The digest embeds a real MRI preview, so do not switch to `0.0.0.0` unless network-wide exposure is explicitly intended and permitted. This simple server has no TLS or authentication. For shared or long-running hosting, place `digest/` behind an authenticated HTTPS-capable web server instead.
