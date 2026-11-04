import time
import httpx
from app.config import settings

def test_all_models():
    api_key = settings.gemini_api_key
    if not api_key:
        print("[!] No GEMINI_API_KEY found in .env")
        return

    print("=" * 80)
    print("  FETCHING AND TESTING ALL MODELS FROM AI STUDIO (https://aistudio.google.com/docs/models)")
    print("=" * 80)

    # 1. Fetch catalog
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    with httpx.Client(timeout=15.0) as client:
        res = client.get(list_url)
        if res.status_code != 200:
            print(f"[!] Failed to fetch models list: {res.status_code}")
            return
        all_models_raw = res.json().get("models", [])

    candidate_models = []
    for m in all_models_raw:
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            candidate_models.append({
                "raw_name": m["name"],
                "clean_name": m["name"].replace("models/", ""),
                "displayName": m.get("displayName", ""),
                "description": m.get("description", "")
            })

    print(f"\nDiscovered {len(candidate_models)} models supporting content generation.\n")

    results = []

    prompt = "Reply with exactly one word: 'CONFIRMED'."

    # 2. Test each model
    for idx, item in enumerate(candidate_models, 1):
        clean_name = item["clean_name"]
        display = item["displayName"]
        print(f"[{idx:02d}/{len(candidate_models):02d}] Testing '{clean_name}' ({display})...", end="", flush=True)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 20}
        }

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=12.0) as client:
                resp = client.post(url, json=payload)

            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

            if resp.status_code == 200:
                body = resp.json()
                text = body.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                status = "SUCCESS"
                note = f"Latency: {elapsed_ms} ms | Output: '{text[:20]}'"
                print(f" -> OK ({elapsed_ms} ms)")
            else:
                status = f"HTTP {resp.status_code}"
                try:
                    err_msg = resp.json().get("error", {}).get("message", resp.text[:60])
                except Exception:
                    err_msg = resp.text[:60]
                note = err_msg.replace("\n", " ")[:70]
                print(f" -> FAILED ({status})")

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            status = "TIMEOUT/ERROR"
            note = str(e)[:60]
            print(f" -> ERROR: {note}")

        results.append({
            "model": clean_name,
            "displayName": display,
            "status": status,
            "latency_ms": elapsed_ms,
            "details": note
        })

    # Summary report
    print("\n" + "=" * 80)
    print("  SUMMARY: GOOGLE AI STUDIO MODEL CAPABILITY MATRIX")
    print("=" * 80)
    
    successful = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] != "SUCCESS"]

    print(f"\n Active & Usable Models: {len(successful)} / {len(results)}")
    for s in sorted(successful, key=lambda x: x["latency_ms"]):
        print(f"  * {s['model']:<35} | {s['latency_ms']:>7} ms | {s['displayName']}")

    print(f"\n Unavailable / Deprecated / Specialized Models: {len(failed)} / {len(results)}")
    for f in failed:
        print(f"  x {f['model']:<35} | {f['status']:<10} | {f['details']}")

if __name__ == "__main__":
    test_all_models()

