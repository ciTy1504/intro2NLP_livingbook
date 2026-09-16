# STEP 2 — Audit of existing projects & environment

Audited 2026-09-16. Everything below was verified by running it, not by reading docs.

## 1. `C:\Users\admin\Desktop\intro-to-nlp` — baseline manuscript

| Property | Value |
|---|---|
| Engine | XeLaTeX (`% !TEX program = xelatex`), `fontspec` + Times New Roman / Arial / Courier New |
| Language | Vietnamese (`\usepackage[vietnamese]{babel}`) |
| Root | `Tex/main.tex` — a flat `\input` manifest in reading order |
| Structure | 2 parts, 15 chapters, 76 sections, 241 subsections, 206 subsubsections |
| Files | 98 `.tex` files under `Tex/part1/`, `Tex/part2/` |
| Size | ~153,000 words / ~11,300 lines |
| Bibliography | `references.bib`, 232 entries, `\bibliographystyle{unsrt}` |
| Citations | 267 `\cite` calls, 221 unique keys, **0 orphans**, 26 uncited entries |
| Labels | 432 (`ssec:` 247, `sec:` 74, `fig:` 54, `sssec:` 20, `chap:` 15, `eq:` 14, `tab:` 8) |
| Images | 62 files in `Tex/images/`, 43 `\includegraphics` + 14 `\bookimage` |
| Build | Verified: XeLaTeX in-place, 297 pages, exit 0 |

### Conventions the Living Book system must preserve

- **Chapter files are `intro.tex`** inside each `chapters_*` directory; sibling files hold sections.
- **`style.tex` custom environments**: `definition`, `example`, `recipe` (tcolorbox theorem-style, each
  taking `{title}{label}`), plus `minted` code blocks requiring `-shell-escape`.
- **`\newtag`** renders a red `[MỚI & SỬA ĐỔI]` marker — the established way to flag new/revised material.
  The Writer Agent uses this for content it adds.
- **`\bookimage[width]{key}{description}`** is the critical hook: it renders the image if
  `images/<key>.png|.jpg` exists, otherwise an orange placeholder box containing the description.
  **This is already a visual-requirement declaration format** — the Visual Engine plugs straight into it.
- **Label namespacing** is consistent (`sec:`, `ssec:`, `fig:`, `eq:`, `tab:`) and must be continued.
- Figures use `\begin{center} ... \captionof{figure}{...} \label{fig:...} \end{center}`, with the caption
  citing the source paper.

### Figure provenance is already a convention

All 15 `\bookimage` keys currently resolve to a file, so there is no figure backlog. (An earlier shell
check in this audit reported 14 missing; that was a bad `sed` capture group. The parser-based check is
authoritative: 15/15 present.)

More importantly, `Tex/images/SOURCES.md` already records, per image: filename, source URL, originating
paper, and licence — including an explicit note that arXiv-sourced figures would need permission for
commercial publication, and flagging the two Nature figures as CC BY 4.0. **The Visual Engine's Asset
Manager must maintain this table rather than invent a parallel one**, and its licence discipline should
match what the author already applies by hand.

`llm_research_reading_list_2026.txt` (a curated, tagged paper list) is a high-quality seed corpus for
Research Memory — copied to `knowledge/seeds/`.

### Additional structure found by the parser

The book also contains `preface.tex`, `ending.tex` and `appendix_reading_map.tex` (an appendix mapping the
2026 reading list onto chapters), each using `\chapter`. Counting those, the parser finds 18 `\chapter`,
85 `\section`, 241 `\subsection`, 214 `\subsubsection` across 99 files and ~143k words of prose
(the earlier 153k figure counted LaTeX markup as words).

## 2. `C:\Users\admin\Desktop\LLM` — Gemini key pool

Relevant file: `cti-realm-sft-final/scripts/common/gemini_pool.py`. A second, simpler variant exists at
`EngChi/data-gen/key_pool.py` (round-robin, 5-min cooldown, env-or-file loading).

**What it does well, and we keep:**

- Random key selection rather than round-robin. The module documents the reason: the pool is shared across
  independent processes whose round-robin cursors all start at 0, so they collide in lockstep. Random
  selection decorrelates them with zero coordination. That reasoning still holds and is preserved.
- Cooldown (not eviction) on 429/403, with a wait-for-soonest-free fallback when every key is cooling.
- Both `AIzaSy...` and `AQ.Ab8...` prefixes are valid `?key=` credentials — **verified independently here**,
  both returned a 58-model `ListModels` response.
- `from_file` reads `utf-8-sig` (the pool file has a BOM) and skips `#` comments.

**What it lacks, and we add:** usage/token tracking, permanent eviction of structurally invalid keys, model
fallback chains, structured output (`responseSchema`), embeddings, a response cache, and a
provider-agnostic interface.

**Key pool file**: `ViettelAILab_keypool.txt`, 254 keys (36 `AIzaSy` + 218 `AQ.Ab8`).
Copied to `secrets/gemini_keypool.txt` and gitignored.

### Live model availability (measured, 10 parallel calls per model, distinct keys)

| Model | Success | Avg latency | Notes |
|---|---|---|---|
| `gemini-3.6-flash` | **10/10** | 1.8s | Reliable workhorse |
| `gemini-3.5-flash-lite` | **10/10** | 1.0s | Fastest |
| `gemini-3-flash-preview` | **10/10** | 1.5s | Reliable |
| `gemini-3.1-flash-lite` | **10/10** | 4.4s | Reliable, slower |
| `gemini-3.5-flash` | 3/10 | 41.4s | Mostly 503 |
| `gemini-3.7-flash` | 0/10 | — | 503 / timeout |
| `gemini-3.8-flash` | 0/10 | — | 503 "high demand" / timeout |
| `gemini-flash-latest` | 0/10 | — | 503 / timeout |
| `gemini-3.1-pro-preview` | **0/10** | — | **429 on every key tried** |
| `gemini-pro-latest` | **0/10** | — | **429 on every key tried** |
| `gemini-2.5-flash` | — | — | 404, retired for new users |
| `gemini-embedding-001` | ok | 0.5s | dim 3072 |
| `gemini-embedding-2` | ok | 0.6s | dim 3072 |

**Design consequence:** no Pro-tier capacity exists on this pool. A single configured model per role would
make the system unusable. The Model Selector therefore resolves a **role to an ordered fallback chain**, and
a 503/429 on one model advances to the next rather than failing the task.

Structured output via `responseMimeType: application/json` + `responseSchema` verified working on
`gemini-3.1-flash-lite`.

## 3. `C:\Users\admin\Desktop\EngChi` — email

`BE/src/services/mailer.js`: nodemailer, `service: 'gmail'`, credentials from `GMAIL_USER` / `GMAIL_PASS`
(a Google App Password), HTML built as a table-based responsive template with inline CSS, sent via
`transporter.sendMail({from, to, subject, html})`.

`BE/src/jobs/reengagement.job.js` contributes the operational pattern: `node-cron` schedule with an explicit
timezone (`Asia/Ho_Chi_Minh`), per-recipient try/catch so one failure does not abort the batch, a small
inter-send delay, and a persisted per-recipient stage counter so the same notification is never re-sent.

**Reused:** the credential model (Gmail + App Password), the inline-CSS HTML template approach, per-send
error isolation, and send-state persistence. **Reimplemented** in Python `smtplib` + `email.message`, since
the Living Book system is Python. No credentials were copied; they go in `secrets/.env`.

Note: `EngChi/gleaming-bus-471709-h3-*.json` is a Google Cloud service-account key committed to that repo.
Not used here, and flagged as worth rotating.

## 4. `C:\Users\admin\Desktop\intro-to-cv` — visual workflow

`gen_images.py` is a single ~180-line matplotlib script: `FancyBboxPatch` boxes and `annotate` arrows laid
out by hand-tuned coordinates, saved to `Tex/images/*.png` at dpi=120. There is no search, no verification,
no licensing and no asset management — it is a one-off generator.

**Reused:** the rendering approach. Box-and-arrow architecture diagrams are exactly what an NLP textbook
needs, matplotlib produces clean deterministic output with no external service, and the figures match the
book's existing look. It becomes the `render_diagram` backend of the Visual Engine, driven by a declarative
spec instead of hand-written coordinates.

**Not reused:** the manual workflow. The Visual Engine adds need detection, licensed image search, semantic
verification, refinement loops, figure QA and asset management around it.

## 5. GitHub repository

`https://github.com/ciTy1504/intro2NLP_livingbook` — exists, public, owner `ciTy1504`,
**zero commits (empty repository)**. `git ls-remote` returns nothing.

Local git identity is already `ciTy1504 <bong1552004@gmail.com>`. No credential helper is configured, so
pushing requires a `GITHUB_TOKEN` in `secrets/.env` (the Git Agent uses it as an HTTPS auth header; it is
never written into the remote URL or any commit).

The old repo `ciTy1504/intro-to-nlp` is **never** used as a remote. The baseline was copied file-by-file
into `manuscript/`, excluding build artifacts, with no `.git` directory carried over.

## 6. Research API availability (all probed live)

| Source | Status | Notes |
|---|---|---|
| arXiv API | works | No key, no auth |
| OpenAlex | works | 26k hits on a test query; `mailto=` polite pool |
| Crossref | works | Journal DOIs + bibliographic search |
| DataCite | works | **Required for arXiv DOIs** — Crossref 404s on `10.48550/*` |
| Unpaywall | works | OA status |
| GitHub API | works | 60 req/h unauthenticated, 5000 with token |
| Hugging Face models/datasets/daily_papers | works | No key |
| OpenReview v2 | works | Conference submissions + reviews |
| ACL Anthology | works | Full 12.6 MB bib dump |
| Hacker News (Algolia) | works | Community signal |
| Wikimedia Commons | works | **Carries license metadata** — primary image source |
| Openverse | works | CC-licensed images with explicit license field |
| RSS (lilianweng, HF blog, …) | works | Research blog agent |
| Semantic Scholar | **429** | Hard-limited unauthenticated; demoted to optional, key-gated |
| Reddit JSON | **403** | Blocked for datacenter/script UAs; disabled by default |
| DuckDuckGo HTML | works via POST | GET returns no results; POST form works |

**Design consequences:** OpenAlex is the primary scholarly backend with Semantic Scholar as optional
enrichment; DOI verification routes by prefix to DataCite or Crossref; community signal comes from Hacker
News + GitHub issues/discussions + HF discussions, with Reddit behind a disabled-by-default flag.

## 7. Environment — installed during this audit

Nothing below was present beforehand; Python was only a Microsoft Store alias stub.

| Component | Version |
|---|---|
| Python | 3.13.15 (+ pip 26.2.1) |
| MiKTeX / XeLaTeX | 25.12 / MiKTeX-XeTeX 4.16, `AutoInstall=1` |
| GitHub CLI | 2.100.0 |
| Pygments (for `minted`) | 2.21.0 |
| Python packages | httpx, pyyaml, python-dotenv, feedparser, beautifulsoup4, lxml, matplotlib, numpy, networkx, pygments, rich, pydantic, jinja2, tenacity, bibtexparser, pymupdf |

**Build verification:** `-output-directory` breaks `minted` v3 (it cannot resolve
`TEXMF_OUTPUT_DIRECTORY`). Building in-place inside a disposable copied workspace succeeds — 297 pages,
exit 0. That is the strategy the `build_book` tool uses, and it also keeps `manuscript/` free of artifacts.
