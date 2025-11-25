"""
Extract Label Studio Access Token from Browser
"""
print("=" * 80)
print("EXTRACT YOUR ACCESS TOKEN FROM BROWSER")
print("=" * 80)
print("""
I can see from the logs that you're accessing /api/token successfully!

Here's how to get your access token:

METHOD 1 - Browser Developer Tools:
1. In Label Studio (http://localhost:8080), go to Account & Settings → Access Token
2. Open Developer Tools (F12)
3. Go to Network tab
4. Refresh the page or click on the Access Token tab
5. Find the request: GET /api/token
6. Click on it
7. Go to "Response" or "Preview" tab
8. You should see JSON like:
   [
     {
       "id": 1,
       "key": "a1b2c3d4e5f6g7h8...",  <-- THIS IS YOUR ACCESS TOKEN
       "user": 1,
       "created": "2025-11-24T..."
     }
   ]
9. Copy the "key" value (the long string without dots)

METHOD 2 - Copy from UI:
- If Label Studio displays the token in the UI, copy it directly
- It should be a long alphanumeric string (40+ characters)
- No dots, no JWT format

Once you have it, share it and I'll update the import script!
""")
print("=" * 80)

