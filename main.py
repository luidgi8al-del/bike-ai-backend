from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Backend Alex fonctionne"}


@app.get("/strava/callback")
def strava_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None)
):
    if error:
        return JSONResponse(
            {"status": "error", "message": error},
            status_code=400
        )

    if not code:
        return JSONResponse(
            {"status": "error", "message": "code manquant"}
        )

    strava_client_id = os.getenv("STRAVA_CLIENT_ID")
    strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")

    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": strava_client_id,
            "client_secret": strava_client_secret,
            "code": code,
            "grant_type": "authorization_code"
        },
        timeout=20
    )

    return JSONResponse(response.json())
