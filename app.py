# Test FAISS loading before deployment

print("\nTesting FAISS loading...")


test_db = FAISS.load_local(
    "university_rag_faiss",
    embedding_model,
    allow_dangerous_deserialization=True
)


results = test_db.similarity_search(
    "What are admission requirements?",
    k=3
)


print("\nRetrieval Test:")

for r in results:
    print(
        r.metadata["source_file"],
        "Page:",
        r.metadata["page_number"]
    )


print("\nFAISS test completed successfully ✅")
