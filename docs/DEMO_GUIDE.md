# A 3–5 minute demonstration

The linked `demo.mp4` is a captioned walkthrough of captured live API runs, including
their real questions, answers, citations, and an investigated retrieval failure.
It is a replay of recorded test evidence, not a screen recording of live typing.
For an instructor-led interactive demonstration, run the application and follow
the sequence below. Never show your `.env` or Streamlit secrets while recording.

1. Open the assistant with `streamlit run streamlet.py`. Explain that this is a
   classroom prototype for sales and support staff at a fictional Pakistani
   menswear retailer. Show the five document names in the sidebar.
2. Ask: “What is the price of the Classic Leather Oxford?” Open Sources and
   Inspect retrieved chunks. Explain that an excerpt is copied from retrieved
   text and that filename labels are added by the backend.
3. Ask: “What does a custom-stitched suit cost, and how much advance do I pay?”
   Show how products and payment/operations evidence combine for the price and
   advance calculation.
4. Ask: “How did the delivery fee and free-delivery minimum change from the
   previous policy to the current policy?” Show both versions. Explain that
   `rag.py` selects eligible policy metadata before semantic and keyword search.
5. Ask: “Do you deliver internationally?” Show the exact insufficient-information
   response. The assistant cannot infer international delivery from Pakistan-wide delivery.
6. Show the two security tests in `evaluation/test_results.csv` and their raw
   contexts in `evaluation/test_results.json`. The malicious visitor note is a
   separate test document and is deliberately inserted into the model context
   to make the test adversarial. It is never shipped as a production business fact.
7. Show the failed/restricted retrieval case in `evaluation/retrieval_experiments.csv`.
   Explain which evidence was absent at k=1 and what changed at k=3 or k=5.
8. For the live challenge, edit one policy value in `data/03_policy_current_v3.txt`,
   save, and click Rebuild knowledge index. Ask again. Explain why changing text
   without re-embedding would leave a vector database stale. Content fingerprints
   automatically select a fresh collection when the app reruns; rebuild is explicit too.

Code ownership prompts: explain `load_chunks`, `select_scope`, `RAG.retrieve`,
the reciprocal-rank formula `1/(60+rank)`, `KnowledgeAgent.ask`, and
`validate_answer`. Explain the difference between checking an exact quote and
proving that a claim follows from it. The evaluation uses a separate model call
and preserves raw evidence; human review remains necessary.
