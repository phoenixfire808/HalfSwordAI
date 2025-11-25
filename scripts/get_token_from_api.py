"""
Get Label Studio Access Token from API
This will help retrieve the actual access token
"""
import requests
import json

LABEL_STUDIO_URL = 'http://localhost:8080'

print("=" * 80)
print("Label Studio Token Retrieval")
print("=" * 80)

# Try to get token using session (if you're logged in via browser)
print("\nMethod 1: Getting token from Label Studio API")
print("Make sure you're logged into Label Studio in your browser first!")
print("\nTo get your access token:")
print("1. Open Label Studio: http://localhost:8080")
print("2. Open browser Developer Tools (F12)")
print("3. Go to Network tab")
print("4. In Label Studio, go to Account & Settings → Access Token")
print("5. Look for the API request to /api/token")
print("6. Click on it and check the Response tab")
print("7. You should see a JSON response with a 'key' field - that's your access token!")
print("\nThe token should look like: a1b2c3d4e5f6... (not a JWT)")
print("=" * 80)

# Alternative: Try to create a new token via API
print("\nMethod 2: Creating a new token via API")
print("If you have your username/password, we can create a token programmatically")
print("=" * 80)

# Show what the token format should be
print("\nExpected Token Format:")
print("- Access Token: Long alphanumeric string (40+ characters)")
print("- Example: 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6'")
print("- NOT a JWT (doesn't have dots like: xxxxx.yyyyy.zzzzz)")
print("=" * 80)

