"""
Get Label Studio API Token
Instructions for obtaining the correct token
"""
print("=" * 80)
print("How to Get Your Label Studio API Token")
print("=" * 80)
print("""
1. Open Label Studio in your browser: http://localhost:8080
2. Log in to your account
3. Click on your profile/account icon (usually top right)
4. Go to "Account & Settings" or "Settings"
5. Look for "Access Token" or "API Token" section
6. Copy the token (it should start with something like: "a1b2c3d4e5f6...")

The token you provided appears to be a refresh token (JWT format).
The Label Studio SDK needs an ACCESS TOKEN, which is usually:
- A shorter alphanumeric string
- Found in Account Settings → Access Token
- Different from the refresh token used for authentication

Once you have the access token, update scripts/import_to_label_studio.py
with the correct token.
""")
print("=" * 80)

