import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def is_url_working(url):
    if not url:
        return False
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def check_urls_multithreaded(urls, max_workers=10):
    results = [False] * len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(is_url_working, url): idx
            for idx, url in enumerate(urls)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
    return results
