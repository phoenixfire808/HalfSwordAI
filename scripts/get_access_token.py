"""
Get Label Studio Access Token
This script helps you retrieve your access token from Label Studio
"""
import requests
import json

LABEL_STUDIO_URL = 'http://localhost:8080'

print("=" * 80)
print("Label Studio Access Token Helper")
print("=" * 80)
print("""
To get your access token:

1. Open Label Studio in your browser: http://localhost:8080
2. Make sure you're logged in
3. Go to: Account & Settings (click your profile icon → Account & Settings)
4. Look for the "Access Token" section
5. You should see your token listed there
6. Click "Copy" or manually copy the token

The token should look like a long alphanumeric string, for example:
a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0...

Once you have the token, you can:
- Update scripts/import_to_label_studio_direct.py with the token
- Or use it directly in API calls

Alternatively, if you just created a token, check the browser's developer console
or network tab to see the token in the API response.
""")
print("=" * 80)

# Try to help them get it programmatically if they're logged in via browser session
print("\nNote: If you're logged into Label Studio in your browser,")
print("you can also check the browser's developer tools:")
print("1. Press F12 to open developer tools")
print("2. Go to Network tab")
print("3. Refresh the page")
print("4. Look for requests to /api/token")
print("5. Check the response - it should contain your token")
print("=" * 80)

