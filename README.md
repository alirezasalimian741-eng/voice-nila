# voice-nila — Persian voice-call server (Groq + Piper Amir)

Single-file Python server: speech-to-text + chat via Groq, speech via local
Piper fa_IR-amir-medium, plus a hold-to-talk call page. Deployed on Render's
free tier from this repo (Docker). Set `GROQ_API_KEY` in the Render dashboard
(environment variables) — never commit a key. Health: `GET /health`.
