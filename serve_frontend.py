import http.server
import socketserver
import os

PORT = 3000
DIRECTORY = "frontend"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Ensure the directory exists
        if not os.path.exists(DIRECTORY):
            os.makedirs(DIRECTORY)
        super().__init__(*args, directory=DIRECTORY, **kwargs)

print(f"\n" + "="*50)
print(f"🚀 FRONTEND SERVER STARTING")
print(f"URL: http://localhost:{PORT}")
print(f"Serving from: ./{DIRECTORY}")
print("="*50 + "\n")

# Use allow_reuse_address to avoid "Address already in use" errors on restart
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend server...")
        httpd.shutdown()
