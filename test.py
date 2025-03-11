import httpx

url = 'http://127.0.0.1:8083/streaming/ask?query=How are you, write in 10 sentences'
with httpx.stream('GET', url) as r:
    for chunk in r.iter_raw():  # or, for line in r.iter_lines():
        print(chunk)