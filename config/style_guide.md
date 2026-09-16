# Style guide for the Writer Agent

Derived from the baseline manuscript. The Writer Agent receives this verbatim. Every rule here was read off
the existing text — the goal is that a new section is indistinguishable from an original one.

## Language and voice

- **The book is written in Vietnamese.** All prose, headings, captions, box titles and list items are
  Vietnamese. Agents reason in English; they write in Vietnamese.
- Technical terms are given in Vietnamese with the English in parentheses on first use in a chapter, then
  the Vietnamese form afterwards. This matches the existing text:
  `Quy luật Mở rộng (Scaling Laws)`, `Lượt xuôi (forward)`, `luật lũy thừa (power law)`.
- Well-established names stay in English and are never translated: Transformer, attention, BERT, GPT, LoRA,
  RAG, MoE, FLOPs, token, embedding, benchmark, dropout, softmax.
- Register is explanatory and confident, addressing the reader as `chúng ta`. Not chatty, not stiff.
- Sentences are complete; avoid telegraphic bullet fragments where a sentence is clearer.

## Structure

- One file per section; the chapter heading lives in that chapter directory's `intro.tex`.
- Every file starts with the two-line header the manuscript already uses:

  ```latex
  % !TEX root = ../main.tex
  % File: part1/chapters_x/secN_name.tex
  ```

- Heading depth: `\section` → `\subsection` → `\subsubsection`. Do not go deeper.
- Label every heading. Namespaces are fixed: `sec:`, `ssec:`, `sssec:`, `fig:`, `eq:`, `tab:`, `chap:`.
  Labels are lowercase ASCII with underscores: `\label{ssec:chinchilla}`.
- New or substantially revised headings carry the existing marker: `\section{Tiêu đề \newtag}`.

## Established environments — use these, do not invent new ones

```latex
\begin{definition}{Tên khái niệm}{def:key}  ... \end{definition}
\begin{example}{Tiêu đề ví dụ}{ex:key}      ... \end{example}
\begin{recipe}{Tiêu đề công thức}{rcp:key}  ... \end{recipe}
```

An "intuition box" opens a hard section, exactly as the pretraining chapter does:

```latex
\begin{tcolorbox}[title={Trực giác cốt lõi}, colback=yellow!10!white,
                  colframe=yellow!50!black, fonttitle=\bfseries]
...
\end{tcolorbox}
```

Code uses `minted` (requires `-shell-escape`, already configured):

```latex
\begin{minted}{python}
...
\end{minted}
```

## Mathematics

- Display equations that are referenced later get `\begin{equation}` + `\label{eq:...}`; otherwise `\[ ... \]`.
- Define every symbol in prose at first use — the existing text always does
  (`Gọi $N$ là số tham số (không tính embedding) và $D$ là số token huấn luyện`).
- Refer to equations with `\eqref{eq:...}`, never a bare number.

## Figures

Never emit an image path directly. Declare the requirement and let the Visual Engine fill it:

```latex
\begin{center}
\bookimage[0.9\textwidth]{khoa_hinh_anh}{Mô tả chi tiết nội dung hình cần có: các thành phần,
quan hệ giữa chúng, và nguồn gốc nếu hình đến từ một bài báo cụ thể.}
\captionof{figure}{Chú thích ngắn gọn giải thích hình. Nguồn: Tác giả et al.~\cite{bibkey}.}
\label{fig:khoa_hinh_anh}
\end{center}
```

The third argument is a full description of what the figure must show — it is both the placeholder text and
the Visual Engine's search specification, so it must be specific about elements and relationships.

## Citations

- `\cite{key}` immediately after the supported statement, before the period.
- When naming the authors in prose, the manuscript uses `Tác giả et al.~\cite{key}` with a non-breaking space.
- **Every** numerical result, benchmark score, SOTA claim, historical date and causal claim carries a
  citation to a primary source.
- Never invent a BibTeX key. Request the citation via the citation pipeline; only `BibtexValidator`
  writes `references.bib`.
- Reuse an existing key when the same source is already cited — check the citation graph first.

## Cross-references

- `\ref{ssec:...}` / `\eqref{eq:...}` / `\ref{fig:...}`, never "the section above".
- Before introducing a concept, check whether the book already defines it and cross-reference instead of
  redefining. Duplicated explanation is a QA failure.

## Scope discipline

- Produce the **minimal coherent patch** that satisfies the verdict. Prefer extending a paragraph over
  adding a subsection, and a subsection over a new section.
- Do not touch content the verdict did not name.
- Do not reformat, re-indent or "tidy" surrounding text — it creates diff noise that hides the real change.
- Indentation is tabs, matching the existing files.
- Keep `\newtag` on new headings so a reader can see what the Living Book added.

## Absolute prohibitions

- Never write a number, benchmark score or date you cannot cite.
- Never assert a technique is state of the art without a source that makes that comparison.
- Never restate an evidence claim as fact when the source only reports someone else's claim
  (this is the citation-laundering pattern the Citation Verifier rejects).
- Never edit `main.tex`, `style.tex` or `references.bib` — other components own those.
