# FaceGuard Pro - Backend Startup Script
# Run this instead of plain `python -m uvicorn main:app --reload`
#
# CRITICAL: --ws-ping-interval 0 --ws-ping-timeout 0
#   Disables WebSocket keepalive pings.
#   Without these, the websockets library fires ping frames every 20 seconds
#   which RACE with our continuous frame sends and cause:
#       AssertionError: assert waiter is None or waiter.cancelled()
#   This crash disconnects all live camera streams.

$env:PYTHONUNBUFFERED = "1"
Set-Location (Join-Path $PSScriptRoot "..")

python -m uvicorn backend.main:app `
    --host 0.0.0.0 `
    --port 8000 `
    --reload `
    --ws-ping-interval 0 `
    --ws-ping-timeout 0 `
    --ws-max-size 16777216 `
    --log-level info
