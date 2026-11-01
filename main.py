# main.py
from src.retrieval.adaptive_retrieval import retrieve
from src.retrieval.trust_scoring import compute_trust
from src.generation.llama_rag import generate_answer
from src.verification.claim_extractor import extract_claims
from src.verification.verifier import verify_claims


def print_banner():
    print("\n" + "=" * 80)
    print("TRUST-AWARE ADAPTIVE RAG")
    print("=" * 80)


def display_documents(documents):
    print("\n4. RETRIEVED AND ACCEPTED EVIDENCE")
    print("-" * 80)

    for idx, doc in enumerate(documents, start=1):
        print(f"\n[Chunk {idx}] Source: {doc['source']}")
        print()

        content = doc["content"]

        print(content[:400])

        if len(content) > 400:
            print("...")


def display_verification(results):

    print("\nClaim Verification Results")
    print("-" * 80)

    supported = 0
    hallucinated = 0

    for idx, item in enumerate(results, start=1):

        claim = item["claim"]
        score = item["score"]
        is_hallucinated = item["hallucinated"]

        status = (
            "HALLUCINATED"
            if is_hallucinated
            else "SUPPORTED"
        )

        print(f"\nClaim {idx}")
        print(f"Text   : {claim}")
        print(f"Score  : {score:.4f}")
        print(f"Status : {status}")

        if is_hallucinated:
            hallucinated += 1
        else:
            supported += 1

    total = len(results)

    if total > 0:
        hallucination_rate = (
            hallucinated / total
        ) * 100
    else:
        hallucination_rate = 0

    print("\n" + "-" * 80)
    print(f"Total Claims         : {total}")
    print(f"Supported Claims     : {supported}")
    print(f"Hallucinated Claims  : {hallucinated}")
    print(
        f"Hallucination Rate   : {hallucination_rate:.2f}%"
    )
    print("-" * 80)


def run_pipeline(query):

    print_banner()

    print("\n1. USER QUERY")
    print("-" * 80)
    print(query)

    print("\n2. PROMPT INSTRUCTIONS")
    print("-" * 80)
    print("• Answer ONLY using the provided context.")
    print("• Do not generate unsupported information.")
    print("• If the answer is unavailable, state that explicitly.")

    print("\n[3] Adaptive Retrieval Started")

    documents = retrieve(query)

    if len(documents) == 0:
        print("No documents retrieved.")
        return

    display_documents(documents)

    print("\n[5] Calculating Trust Score")

    trust_score = compute_trust(
        documents,
        query=query,
        source_type="academic"
    )

    print(
        f"Trust Score: {trust_score:.4f}"
    )

    trust_threshold = 0.50

    if trust_score < trust_threshold:

        print(
            "\nContext trust is too low."
        )

        print(
            "Generation stopped."
        )

        return

    print(
        "\n[6] Building Context"
    )

    context = "\n\n".join(
        doc["content"]
        for doc in documents[:3]
    )

    print(
        "\n[5] Generating Answer using Llama"
    )

    answer = generate_answer(
        query=query,
        context=context
    )

    print("\n5. GENERATED ANSWER (Llama 3.1 8B via Ollama)")
    print("-" * 80)
    print(answer)

    print(
        "\n[6] Extracting Claims"
    )

    claims = extract_claims(
        answer
    )

    print(
        f"Claims Extracted: {len(claims)}"
    )

    for idx, claim in enumerate(
            claims,
            start=1
    ):
        print(f"{idx}. {claim}")

    print(
        "\n[7] Verifying Claims"
    )

    verification_results = verify_claims(
        claims,
        documents
    )

    display_verification(
        verification_results
    )

    print(
        "\n[8] Final Decision"
    )

    hallucinated_count = sum(
        item["hallucinated"]
        for item in verification_results
    )

    if hallucinated_count == 0:

        print(
            "Response appears fully supported."
        )

    elif hallucinated_count <= 2:

        print(
            "Response contains minor hallucinations."
        )

    else:

        print(
            "Response contains significant hallucinations."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    ).strip()

    run_pipeline(query)