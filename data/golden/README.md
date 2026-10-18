# Golden set: annotation guide

`milestone1.yaml` is the ground truth every retrieval number is computed against.
A wrong label makes the report wrong while the code still passes, so the rules
below matter more than the code does.

## How the file is built

The questions were proposed with language-model assistance from the ingested
chunk text, then each proposal was checked against the chunk it cites. Every
labelled chunk was confirmed to contain the fact it is labelled for, and every
labelled chunk id was confirmed to exist in `data/chunks/chunks.jsonl`. That
check is mechanical; judging whether a question is fair and phrased the way a
user would ask it is a human job and remains open.

## What counts as a correct chunk

A chunk is relevant when it contains the answer, meaning the provision, the
figure, or the named item needed to answer the question. A chunk is **not**
relevant merely because it is about the same subject. If a question asks for the
revised Bank Rate, the chunk stating the rate is relevant; the circular
announcing the monetary policy statement is not, unless it also states the rate.

Label every chunk that contains part of the answer and no others. Do not pad a
label set to make a question look easier, and do not trim it to make the numbers
look better.

## Category definitions

| category | meaning |
| --- | --- |
| `single_clause` | one chunk answers it |
| `multi_clause` | two or more chunks of the same document are both needed |
| `cross_document` | the answer needs chunks from different documents |
| `temporal` | the answer depends on which circular is currently in force |
| `unanswerable` | the corpus contains no chunk from which the answer follows |

## How granular a citation must be

At chunk granularity: `doc_id#ordinal`, for example `rbi:13736#0000`. The corpus
is small and chunk ids are stable for a given source snapshot, so no coarser
unit is needed. Chunk ids change when the corpus is re-ingested with a different
chunking strategy; when that happens the evaluation fails on a missing chunk id
on purpose, and the affected questions must be re-verified rather than the whole
set regenerated.

## Overlap caveat

Chunking is fixed size with a 200-character overlap, so a provision that
straddles a boundary appears in two adjacent chunks. When a question's answer
sits in such a region, both chunks contain it and both are labelled. This is
honest but it does mean recall is measured against a slightly generous label
set for those questions; the alternative, labelling one of two chunks that both
contain the answer, would mark a correct retrieval as a miss.

## Unanswerable questions

A question is unanswerable when no chunk in the corpus yields the answer. The
two in this set are deliberately outside the corpus's subject matter: this
corpus is RBI notifications, so questions about GST rates and about the
Income-tax Act cannot be answered from it.

Unanswerable questions have an empty `relevant_chunk_ids`. They score zero on
every retrieval metric by construction, which means the aggregate figures in
`results.md` understate retrieval quality on the questions that can be answered.
That is a limitation of a retrieval-only milestone: there is no abstention
mechanism yet, so there is nothing to score. Abstention, and reporting
unanswerable questions separately, arrives with answer generation.
