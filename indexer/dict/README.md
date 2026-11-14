# Label dictionaries, one file per language

`<lang>.json`: `{ "entries": { "<label as printed>": {"type", "full", "src"} } }`. `type` is one of the room types of
`docs/contract.md` (bedroom, kitchen, bathroom, wc, sauna, balcony, corridor, living, storage, technical, office, other).
A label is matched whole, after lower-casing, removing a trailing room number and folding accents ("MH2" -> "mh").
Whatever is not found here falls through to the word-root rules in `labels.py`.

`src` says how much to trust an entry:

| src | meaning |
|---|---|
| `legend: ...` | taken from a legend printed on a real plan or a guide (the page is named) |
| `real plan: ...` | label seen on a real plan, meaning taken from the full word next to it |
| `dataset` | label that occurs in the preview set, meaning from the full word |
| `knowledge` | standard usage that was not checked against any legend: verify before relying on it |

Abbreviations differ from office to office; the legend of the plan at hand is the only authority.
Finnish legend: Helsinki Ruska, "Asuntopohjissa käytetyt tilanimet".
