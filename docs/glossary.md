# Glossary

Terms used in this project's code, docs and evaluation reports. Two groups: Indian legal and
regulatory vocabulary, then retrieval and evaluation vocabulary.

## Indian legal and regulatory

| Term | Meaning |
|---|---|
| **Act** | A law passed by a legislature. The CGST Act, 2017 is the central statute for GST on intra-state supply. |
| **Bare act** | The text of an act as enacted, without commentary or annotation. |
| **CBIC** | Central Board of Indirect Taxes and Customs. Administers GST, customs and central excise. Publishes circulars and notifications. |
| **CBDT** | Central Board of Direct Taxes. Administers direct taxes. A different board from CBIC, which matters when a question spans both. |
| **CGST** | Central Goods and Services Tax. The central government's share of GST on an intra-state supply. |
| **SGST** | State Goods and Services Tax. The state's share of the same supply. |
| **IGST** | Integrated Goods and Services Tax. Levied on inter-state supply and imports, collected by the Centre and apportioned to states. |
| **Circular** | A regulator's instruction clarifying how a provision applies in practice, or directing officers. Clarifies the law; it does not itself make law. |
| **Explanation** | A clause appended to a section to clarify its meaning. Frequently carries the part that decides an answer. |
| **FEMA** | Foreign Exchange Management Act, 1999. Governs cross-border payments, remittances and foreign investment. |
| **Gazette** | The Gazette of India, the official publication where notifications acquire legal force. |
| **ITC** | Input Tax Credit. Credit for tax already paid on inputs, set against output tax liability. Heavily circular-dependent. |
| **Master Direction** | An RBI document consolidating all instructions on one subject into a single reference, replacing the circulars it supersedes. |
| **Notification** | A formal instrument published in the Gazette that brings a provision, rate or rule into force, or amends one. |
| **Proviso** | A clause within a section that carves out an exception to the main provision. Easy for retrieval to miss, since it can reverse the answer. |
| **Rule** | A subordinate instrument made under an act, typically covering procedure. The CGST Rules, 2017 sit under the CGST Act. |
| **Section / sub-section / clause / sub-clause** | The standard nesting inside an act: section 16, sub-section (2), clause (b), sub-clause (i). |
| **Sub-rule** | The equivalent nesting inside a rule. |
| **Suo motu** | On the body's own motion, without a complaint. Appears in enforcement material. |

### One correctness point worth remembering

Consolidated PDFs published on the CBIC website carry an explicit disclaimer that they are prepared
"for convenience and easy reference" and have **no legal binding or force**. Only notifications as
published in the Gazette of India have the force of law. Any answer this system produces is
therefore a retrieval aid over convenience copies, not legal advice, and the README says so.

## Retrieval and evaluation

| Term | Meaning |
|---|---|
| **Ablation** | Changing or removing one component to measure what it contributes. The results table is a series of ablations. |
| **BM25** | A lexical ranking function based on term frequency and inverse document frequency, with document-length normalisation. Strong on exact terms. |
| **bge** | BAAI General Embedding, a family of open embedding and reranking models used locally here. |
| **Chunk** | An indexed unit of text. The unit of retrieval, and the unit a citation points at. |
| **Chunk id** | Stable identifier for a chunk, derived from document id and ordinal. Golden-set labels reference these. |
| **Cross-encoder** | A model that scores a query and document together, so it sees both at once. More accurate and far slower than embedding similarity, which is why it reranks a shortlist. |
| **Dense retrieval** | Finding documents by nearest neighbours in embedding space. Good at meaning, weak on exact identifiers. |
| **Embedding** | A fixed-length numeric vector representing text, positioned so similar meanings land close together. |
| **Faithfulness** | Whether every claim in an answer is supported by the retrieved context. An answer can be faithful and wrong, or correct and unfaithful. |
| **Golden set** | The hand-verified question set used as ground truth. Its quality bounds the credibility of every number reported. |
| **Hybrid search** | Combining a dense ranked list with a lexical ranked list, then fusing them. |
| **LLM as judge** | Using a language model to score answers against a rubric. Only meaningful once calibrated against human labels. |
| **MRR** | Mean Reciprocal Rank. The average of `1 / rank` of the first relevant result. Punishes a relevant result sitting low in the list. |
| **nDCG@k** | Normalised discounted cumulative gain at k. Rewards relevant results near the top; allows more than one relevant result per question. |
| **precision@k** | Fraction of the top k results that are relevant. Divides by k, not by the number of relevant items. |
| **Qdrant** | The vector database used here, in local on-disk mode. |
| **RAG** | Retrieval-augmented generation. Retrieve evidence, then generate an answer grounded in it. |
| **recall@k** | Fraction of all relevant items that appear in the top k. Divides by the number of relevant items. |
| **Refusal** | Declining to answer when retrieval is weak. A measurable behaviour, scored on deliberately unanswerable questions. |
| **Reranker** | Second-stage scorer applied to a shortlist from first-stage retrieval. |
| **Retriever protocol** | The one-method interface (`retrieve(query, k)`) that every retrieval configuration implements, so evaluation code never changes. |
| **RRF** | Reciprocal Rank Fusion. Merges ranked lists by summing `1 / (k + rank)` per document. Rank-based, so it needs no score calibration between lists. |
| **Temporal filtering** | Restricting retrieval to the version of a provision in force on a given date. The project's intended differentiator. |
| **Vector store** | Storage and search for embeddings. |
