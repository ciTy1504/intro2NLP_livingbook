# Status page

One self-contained HTML file with its data embedded, rebuilt from the live database:

```bash
python -m livingbook.cli dashboard --open
```

Because it carries its own data, the same file works three ways with no backend:

| Where | How |
|---|---|
| Locally | open `dashboard/index.html` — no server, works offline |
| Firebase Hosting | `firebase login && firebase use --add && firebase deploy --only hosting` |
| Claude artifact | published from this session; republish to refresh it |

`template.html` is the source; `index.html` is generated and gitignored. Rebuild it
before deploying — it is a snapshot, not a live view.

The page carries no outbound links and fetches nothing at runtime; drill-down belongs
to the CLI and the MCP tools, which return the whole provenance chain rather than a
bare URL.
