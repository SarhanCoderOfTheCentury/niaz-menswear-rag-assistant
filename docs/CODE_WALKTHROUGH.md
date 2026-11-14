# Explain the application in your own words

The frontend is `streamlet.py` (the filename requested for this project). The
command is `streamlit run streamlet.py`: Streamlit is the framework name.

## Follow one question

1. Streamlit reads `GEMINI_API_KEY` from the environment or its secrets manager.
   A password input is available when neither is configured. `make_agent` caches
   the backend by settings and source fingerprint. A question is validated before
   an API call, so empty input does not spend quota.
2. `RAG` calls `load_chunks`. Each file's simple front matter becomes metadata.
   TXT/Markdown are read directly; text-based PDF pages are extracted using
   `pypdf`. Policy metadata must include version, effective date and status.
3. Headings separate sections. LangChain's recursive splitter divides long
   sections with overlap. Every piece receives a deterministic ID, filename,
   section and policy metadata. Product names/codes remain with their details.
4. The corpus fingerprint includes the text, metadata, splitter settings,
   embedding model and dimensions. It determines the Chroma collection name.
   Editing a source therefore creates a different identity; it cannot silently
   query the previous collection after the UI reruns.
5. `ensure_index` reuses a complete collection or generates document embeddings
   using Gemini. The provider normalizes vectors and caches them by exact text,
   model, dimensions and retrieval task. Query embeddings use a separate task
   type and cache key. All values use the same embedding space.
6. `select_scope` recognizes current, archived, comparison and ISO-date requests.
   `allowed` selects eligible IDs. **Both** Chroma and BM25 see those IDs. Current
   rules cannot accidentally retrieve archived policy text via the keyword path.
7. Semantic search finds meaning matches. BM25 finds literal terms and product
   codes. Hybrid combines ranks with `1/(60+rank)`, not raw scores. A comparison
   reserves evidence from both versions if the context budget allows it.
8. `KnowledgeAgent.ask` sends the original question, selected scope and actual
   evidence to Gemini. System instructions say the documents are untrusted data,
   so a visitor note cannot override the assistant's role.
9. A Pydantic schema requires supported claims with chunk IDs and verbatim
   excerpts. `validate_answer` checks IDs and quote matches. Filenames come from
   metadata. A broken citation causes refusal; it is never shown as trustworthy.
10. The frontend shows the answer, citation labels, filenames, excerpts and full
    retrieved chunks. The full retrieval can include unused evidence. It does
    not mean every retrieved item supports the answer.

## What is and is not proved

Exact quote matching proves that the quoted text was retrieved; it cannot prove
the claim is logically implied by the quote. Likewise, a prompt saying “use only
context” does not prove reliability. `evaluate.py` checks actual generated
answers with a separate grading call and deterministic source/refusal/injection
checks. Full evidence is saved for a reviewer to inspect. Automated grading can
still be wrong.

A security test only counts when its attack actually enters the answer model's
context. The harness indexes a separate fixture and deliberately includes that
chunk along with valid evidence, even if ordinary search would miss the attack.
The CSV labels this as an injection test and the raw evidence records insertion.

No expected test answer is looked up by the application. Test expectations exist
only in `evaluation/test_cases.json` and the evaluator. Runtime answers always
come from the retrieved documents and Gemini.

## Live challenge

Explain the difference between editing a TXT file and updating a vector database.
The former changes the source; the latter needs new chunks/vectors. Click Rebuild
knowledge index to make this explicit. Fingerprints also select a fresh collection
automatically when the source changes. `scripts/check_live_update.py` demonstrates
this with actual Gemini calls on a temporary copy, keeping the original documents
unchanged.

## Boundaries

This is an evidence-grounded Q&A agent, with no autonomous order placement,
customer records, web search or payment tools. Dates outside the documented
patterns need explicit wording or a selected policy scope. Free-tier quotas and
model access depend on the user's Gemini project. API failures are displayed and
never converted into invented successful test rows.
