"""
Find your Label Studio Access Token
"""
print("=" * 80)
print("HOW TO GET YOUR LABEL STUDIO ACCESS TOKEN")
print("=" * 80)
print("""
The token you provided is a REFRESH TOKEN (JWT format).
Label Studio API needs an ACCESS TOKEN (simple alphanumeric string).

STEP-BY-STEP INSTRUCTIONS:

1. Open Label Studio in your browser: http://localhost:8080
   (Make sure you're logged in)

2. Open Browser Developer Tools:
   - Press F12, or
   - Right-click → Inspect → Network tab

3. In Label Studio:
   - Click your profile icon (top right)
   - Go to "Account & Settings"
   - Click on "Access Token" tab

4. In Developer Tools Network tab:
   - Look for a request to: /api/token
   - Click on it
   - Go to "Response" or "Preview" tab
   - You should see JSON like:
     {
       "id": 1,
       "key": "a1b2c3d4e5f6g7h8i9j0...",  <-- THIS IS YOUR ACCESS TOKEN
       "user": 1,
       "created": "2025-11-24T..."
     }

5. Copy the "key" value - that's your access token!

ALTERNATIVE METHOD:

If you see the token displayed in the Label Studio UI:
- It should be a long string without dots
- Example: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0...
- Copy that entire string

Once you have the access token, share it and I'll update the import script!
""")
print("=" * 80)

