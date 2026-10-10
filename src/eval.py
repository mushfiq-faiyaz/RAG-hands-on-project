import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pipeline import ask


RANK1_TESTS = [
    ("What is the latest version of PidiFie?", "RanFy_Monthly_Oct2026.pdf", 1),
]


TESTS = [
    ("How much does Fresh Texfy Pro cost?", ["5.99"]),
    ("Can I get my money back?", ["7 days", "14 days"]),
    ("When is support available?", ["Sunday", "Thursday"]),
    ("What is the latest version of PidiFie?", ["1.2"]),
    ("Who founded RanFy and when?", ["Mushfiqur", "14 March 2025"]),
]

passed = 0
for question, expected_terms in TESTS:
    result = ask(question, [])
    # answer = result["answer"].lower()
    answer = result["answer"].lower()
    answer = answer.replace("\u202f", " ").replace("\u00a0", " ")
    answer = " ".join(answer.split())
    ok = all(term.lower() in answer for term in expected_terms)
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"[{status}] {question}")
    if not ok:
        print(f"   expected to contain: {expected_terms}")
        print(f"   got: {result['answer'][:120]}")

print(f"\n{passed}/{len(TESTS)} passed")





print("\nRank-1 checks:")
for question, expected_source, expected_page in RANK1_TESTS:
    result = ask(question, [])
    top = result["sources"][0]
    ok = top["source"] == expected_source and top["page"] == expected_page
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {question}")
    print(f"   expected top-1: {expected_source} p.{expected_page}")
    print(f"   got top-1:      {top['source']} p.{top['page']}")