curl -X POST http://127.0.0.1:8000/ask ^
     -H "Content-Type: application/json" ^
     -d "{\"query_text\": \"Jika aku suka matematika dan fisika, enaknya kau masuk jurusan apa?\"}" ^
     -o response.txt

