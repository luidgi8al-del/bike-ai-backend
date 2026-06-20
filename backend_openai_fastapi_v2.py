from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY manquante. Ajoute-la dans .env ou dans les variables d'environnement."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Coach Cyclisme IA V2", version="2.1.0")


# ─────────────────────────────────────────────
#  Modèles communs
# ─────────────────────────────────────────────

class ProfilePayload(BaseModel):
    sport: str = "cyclisme"
    discipline: Optional[str] = None
    objectif: Optional[str] = None
    ftp: Optional[int] = None
    watts_5min: Optional[int] = None
    fc_max: Optional[int] = None
    fc_repos: Optional[int] = None
    fc_repos_profil: Optional[int] = None
    poids_kg: Optional[float] = None


class ActivityItem(BaseModel):
    type: Optional[str] = None
    contenu: Optional[str] = None
    duree_min: Optional[int] = None
    rpe: Optional[int] = None
    commentaire: Optional[str] = None
    watts_moyens: Optional[int] = None
    fc_moyenne: Optional[int] = None
    distance_km: Optional[float] = None
    statut: Optional[str] = None


class DayPayload(BaseModel):
    charge_jour: Optional[int] = None
    charge_7j: Optional[int] = None
    charge_28j: Optional[int] = None
    ctl: Optional[float] = None
    atl: Optional[float] = None
    tsb: Optional[float] = None
    ratio_fatigue_forme: Optional[float] = None
    fatigue_subjective: Optional[str] = None
    envie_du_jour: Optional[str] = None
    temps_disponible_min: Optional[int] = None
    hrv: Optional[Any] = None
    fc_repos_tendance: Optional[Any] = None
    jours_intenses_recents: Optional[int] = None
    activites_du_jour: List[ActivityItem] = Field(default_factory=list)
    historique_recent: List[str] = Field(default_factory=list)
    sante: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────
#  Modèles /coach/recommendation
# ─────────────────────────────────────────────

class CoachRequest(BaseModel):
    message: str
    profile: ProfilePayload
    day_data: DayPayload


class CoachResponse(BaseModel):
    resume: str
    niveau_alerte: Literal["vert", "bleu", "orange", "rouge"]
    discipline: str
    objectif_du_jour: str
    type_seance: str
    duree_totale_min: int
    echauffement: str
    bloc_principal: str
    retour_au_calme: str
    intensite: str
    pourquoi: str
    vigilance: str
    alternative: str
    rpe_estime: int
    tags: List[str]
    moteur: str = "openai"


# ─────────────────────────────────────────────
#  Modèles /coach/dashboard
# ─────────────────────────────────────────────

class DashboardRequest(BaseModel):
    mode: str = "dashboard_analysis"
    profile: ProfilePayload
    day_data: DayPayload


class DashboardResponse(BaseModel):
    titre: str
    analyse: str
    niveau_alerte: Literal["vert", "bleu", "orange", "rouge"]
    conseil_court: str


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def build_context_summary(req: CoachRequest) -> Dict[str, Any]:
    p = req.profile
    d = req.day_data
    discipline = (p.discipline or "").strip().lower()
    objectif = (p.objectif or "").strip()

    fatigue_flag = "normale"
    if (d.tsb is not None and d.tsb <= -15) or (d.jours_intenses_recents or 0) >= 3:
        fatigue_flag = "élevée"
    elif (d.tsb is not None and d.tsb <= -7) or (d.jours_intenses_recents or 0) >= 2:
        fatigue_flag = "modérée"
    elif (d.tsb is not None and d.tsb >= 5):
        fatigue_flag = "fraîcheur élevée"

    specificity = []
    if discipline in {"xco", "vtt", "cross-country"}:
        specificity.append("travail de relances, PMA courte, VO2max spécifique XCO, variations de cadence")
    if discipline in {"route", "vélo route"}:
        specificity.append("travail au seuil, tempo, VO2, endurance, sprints selon contexte")
    if discipline in {"home trainer", "home-trainer", "ht"}:
        specificity.append("formats très structurés et précis en temps et intensité")

    return {
        "fatigue_flag": fatigue_flag,
        "discipline_normalisee": discipline or "cyclisme",
        "objectif": objectif or "performance générale",
        "specificity": specificity,
    }


def build_health_summary(sante: Optional[Dict[str, Any]]) -> str:
    if not sante:
        return "Aucune donnée santé disponible."
    parts = []
    fc = sante.get("fc_repos_mesure")
    if fc:
        parts.append(f"FC repos mesurée ce matin : {fc} bpm")
    sommeil_h = sante.get("sommeil_heures")
    if sommeil_h:
        parts.append(f"Durée de sommeil dernière nuit : {sommeil_h:.1f}h")
    phases = sante.get("sommeil_phases")
    if phases and isinstance(phases, dict):
        phases_str = ", ".join(f"{k} {v} min" for k, v in phases.items() if v)
        if phases_str:
            parts.append(f"Phases de sommeil : {phases_str}")
    return ". ".join(parts) + "." if parts else "Aucune donnée santé disponible."


# ─────────────────────────────────────────────
#  Schémas JSON stricts
# ─────────────────────────────────────────────

RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "coach_response",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "resume": {"type": "string"},
            "niveau_alerte": {"type": "string", "enum": ["vert", "bleu", "orange", "rouge"]},
            "discipline": {"type": "string"},
            "objectif_du_jour": {"type": "string"},
            "type_seance": {"type": "string"},
            "duree_totale_min": {"type": "integer"},
            "echauffement": {"type": "string"},
            "bloc_principal": {"type": "string"},
            "retour_au_calme": {"type": "string"},
            "intensite": {"type": "string"},
            "pourquoi": {"type": "string"},
            "vigilance": {"type": "string"},
            "alternative": {"type": "string"},
            "rpe_estime": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": [
            "resume", "niveau_alerte", "discipline", "objectif_du_jour", "type_seance",
            "duree_totale_min", "echauffement", "bloc_principal", "retour_au_calme",
            "intensite", "pourquoi", "vigilance", "alternative", "rpe_estime", "tags"
        ]
    },
    "strict": True
}

DASHBOARD_JSON_SCHEMA: Dict[str, Any] = {
    "name": "dashboard_response",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "titre": {"type": "string"},
            "analyse": {"type": "string"},
            "niveau_alerte": {"type": "string", "enum": ["vert", "bleu", "orange", "rouge"]},
            "conseil_court": {"type": "string"}
        },
        "required": ["titre", "analyse", "niveau_alerte", "conseil_court"]
    },
    "strict": True
}


# ─────────────────────────────────────────────
#  Prompts système
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
Tu es un coach cycliste expert premium, précis, prudent et orienté performance.
Tu réponds toujours en français.
Tu analyses le contexte à partir du profil, de la charge, de la fatigue, du temps disponible, de la discipline et du message libre.

Références méthodologiques :
- Coggan : logique FTP, PMA/VO2, zones de puissance, charge aiguë/chronique, CTL / ATL / TSB.
- Rønnestad : formats d'intervalles courts/fragmentés VO2max pertinents selon fraîcheur et niveau.
- Spécificité cyclisme : différencier route, VTT, XCO, home trainer.
- Toujours adapter la séance au niveau de fatigue et au temps disponible.

Règles absolues :
1. Ne jamais inventer des données absentes.
2. S'il y a fatigue élevée, TSB très négatif, plusieurs jours intenses ou récupération insuffisante : réduire le volume.
3. Si l'utilisateur veut une séance intense mais que l'état du jour n'est pas favorable : expliquer et proposer une alternative.
4. Pour VTT : relances, PMA, VO2max, changements de rythme, départs explosifs.
5. Pour route : endurance, tempo, seuil, VO2max, sprint, sortie longue.
6. Prescriptions exploitables : durée, logique d'effort, récupération, repères d'intensité.
7. Si FTP disponible, utiliser des plages en % FTP. Sinon, RPE et sensations.
8. Toujours proposer une alternative plus facile.
9. La séance doit être réaliste pour le temps disponible.

Repères TSB :
- TSB <= -15 : alerte fatigue forte
- TSB -7 à -14 : fatigue modérée
- TSB -6 à +4 : zone exploitable
- TSB > +5 : fraîcheur intéressante
"""

DASHBOARD_SYSTEM_PROMPT = """
Tu es un coach cycliste expert. Tu analyses l'état de forme du jour d'un athlète cycliste
et tu fournis une analyse courte, personnalisée et actionnable en français.

Tu disposes des données : charge d'entraînement (CTL, ATL, TSB), historique récent des séances,
profil (FTP, FC max), et données santé (FC repos mesurée, durée et phases de sommeil).

Règles :
1. Synthétise TOUTES les données disponibles — ne te limite pas au TSB.
2. FC repos mesurée : si anormalement élevée par rapport au profil, c'est un signe de fatigue ou d'infection.
3. Sommeil court (< 6h) = signal négatif fort. Sommeil long avec bon profond/REM = signal positif.
4. Sois concis et utile. "analyse" = 2 à 4 phrases maximum.
5. "conseil_court" = une action concrète en 1 phrase.
6. "niveau_alerte" : vert = super forme, bleu = équilibre, orange = fatigue modérée, rouge = fatigue forte.
7. Ne jamais inventer de données. Si absente, ignore-la.
8. Réponds strictement en JSON selon le schéma.
"""


# ─────────────────────────────────────────────
#  Logique OpenAI
# ─────────────────────────────────────────────

def openai_recommendation(req: CoachRequest) -> CoachResponse:
    context_summary = build_context_summary(req)

    user_payload = {
        "message_utilisateur": req.message,
        "profil": req.profile.model_dump(),
        "donnees_du_jour": req.day_data.model_dump(),
        "analyse_interne": context_summary,
    }

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyse ce contexte et réponds au format JSON strict demandé.\n\n"
                    + json.dumps(user_payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": RESPONSE_JSON_SCHEMA["name"],
                "schema": RESPONSE_JSON_SCHEMA["schema"],
                "strict": RESPONSE_JSON_SCHEMA["strict"],
            }
        },
    )

    raw_text = getattr(response, "output_text", None)
    if not raw_text:
        raise HTTPException(status_code=500, detail="Réponse OpenAI vide.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"JSON OpenAI invalide : {exc}") from exc

    try:
        return CoachResponse(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Réponse OpenAI incomplète : {exc}") from exc


def openai_dashboard(req: DashboardRequest) -> DashboardResponse:
    d = req.day_data
    p = req.profile
    health_summary = build_health_summary(d.sante)

    user_payload = {
        "profil": {
            "ftp": p.ftp or p.watts_5min,
            "fc_max": p.fc_max,
            "fc_repos_reference": p.fc_repos or p.fc_repos_profil,
            "poids_kg": p.poids_kg,
        },
        "charge_entrainement": {
            "ctl": d.ctl,
            "atl": d.atl,
            "tsb": d.tsb,
            "charge_7j": d.charge_7j,
            "charge_28j": d.charge_28j,
            "ratio_fatigue_forme": d.ratio_fatigue_forme,
            "jours_intenses_recents": d.jours_intenses_recents,
        },
        "donnees_sante": health_summary,
        "historique_recent": d.historique_recent[:5],
    }

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": DASHBOARD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyse l'état de forme de cet athlète aujourd'hui et réponds en JSON strict.\n\n"
                    + json.dumps(user_payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": DASHBOARD_JSON_SCHEMA["name"],
                "schema": DASHBOARD_JSON_SCHEMA["schema"],
                "strict": DASHBOARD_JSON_SCHEMA["strict"],
            }
        },
    )

    raw_text = getattr(response, "output_text", None)
    if not raw_text:
        raise HTTPException(status_code=500, detail="Réponse OpenAI dashboard vide.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"JSON dashboard invalide : {exc}") from exc

    try:
        return DashboardResponse(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Réponse dashboard incomplète : {exc}") from exc


# ─────────────────────────────────────────────
#  Routes existantes
# ─────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "api_key_loaded": bool(OPENAI_API_KEY),
    }


@app.post("/coach/recommendation", response_model=CoachResponse)
def coach_recommendation(req: CoachRequest) -> CoachResponse:
    return openai_recommendation(req)


@app.post("/coach/dashboard", response_model=DashboardResponse)
def coach_dashboard(req: DashboardRequest) -> DashboardResponse:
    """
    Analyse de forme du jour pour le tableau de bord.
    Prend en compte CTL/ATL/TSB, FC repos mesurée, sommeil (durée + phases), historique récent.
    """
    return openai_dashboard(req)


@app.get("/strava/callback")
def strava_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None)
):
    if error:
        return JSONResponse({"status": "error", "message": error}, status_code=400)

    if not code:
        return JSONResponse({"status": "error", "message": "code manquant"})

    strava_client_id = os.getenv("STRAVA_CLIENT_ID")
    strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")

    if not strava_client_id or not strava_client_secret:
        return JSONResponse(
            {"status": "error", "message": "variables Strava manquantes côté backend"},
            status_code=500
        )

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

    token_data = response.json()

    redirect_url = (
        f"alexapp://strava-callback"
        f"?status=success"
        f"&access_token={token_data.get('access_token')}"
        f"&refresh_token={token_data.get('refresh_token')}"
        f"&expires_at={token_data.get('expires_at')}"
    )

    return RedirectResponse(url=redirect_url)


@app.post("/strava/refresh")
def strava_refresh(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rafraîchit un token Strava expiré.
    """
    refresh_token = body.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token manquant")

    strava_client_id = os.getenv("STRAVA_CLIENT_ID")
    strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")

    if not strava_client_id or not strava_client_secret:
        raise HTTPException(status_code=500, detail="Variables Strava manquantes côté backend")

    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": strava_client_id,
            "client_secret": strava_client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        },
        timeout=20
    )

    if not response.ok:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Erreur Strava : {response.text}"
        )

    data = response.json()
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "expires_at": data.get("expires_at"),
    }


# ─────────────────────────────────────────────
#  Route Garmin Upload
# ─────────────────────────────────────────────

GARMIN_SSO_URL = "https://sso.garmin.com/sso/signin"
GARMIN_UPLOAD_URL = "https://connect.garmin.com/modern/proxy/upload-service/upload/.tcx"

def garmin_login(email: str, password: str) -> requests.Session:
    """
    Authentification Garmin Connect via SSO.
    Retourne une session authentifiée.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "origin": "https://sso.garmin.com",
        "referer": "https://sso.garmin.com/sso/signin",
    })

    # 1. Récupérer le ticket SSO
    params = {
        "service": "https://connect.garmin.com/modern/",
        "webhost": "https://connect.garmin.com",
        "source": "https://connect.garmin.com/signin/",
        "redirectAfterAccountLoginUrl": "https://connect.garmin.com/modern/",
        "redirectAfterAccountCreationUrl": "https://connect.garmin.com/modern/",
        "gauthHost": "https://sso.garmin.com/sso",
        "locale": "fr_FR",
        "id": "gauth-widget",
        "cssUrl": "https://static.garmincdn.com/com.garmin.connect/ui/css/gauth-custom-v1.2-min.css",
        "clientId": "GarminConnect",
        "rememberMeShown": "true",
        "rememberMeChecked": "false",
        "createAccountShown": "true",
        "openCreateAccount": "false",
        "displayNameShown": "false",
        "consumeServiceTicket": "false",
        "initialFocus": "true",
        "embedWidget": "false",
        "generateExtraServiceTicket": "true",
        "generateTwoExtraServiceTickets": "false",
        "generateNoServiceTicket": "false",
        "globalOptInShown": "true",
        "globalOptInChecked": "false",
        "mobile": "false",
        "connectLegalTerms": "true",
        "showTermsOfUse": "false",
        "showPrivacyPolicy": "false",
        "showConnectLegalAge": "false",
        "locationPromptShown": "true",
        "showPassword": "true",
        "useCustomHeader": "false",
        "mfaRequired": "false",
        "performMFACheck": "false",
        "checkPassword": "false",
    }

    get_resp = session.get(GARMIN_SSO_URL, params=params, timeout=15)
    if not get_resp.ok:
        raise HTTPException(status_code=502, detail="Impossible de contacter Garmin SSO.")

    # 2. POST credentials
    post_data = {
        "username": email,
        "password": password,
        "embed": "false",
        "_eventId": "submit",
        "displayNameRequired": "false",
    }

    post_resp = session.post(
        GARMIN_SSO_URL,
        params=params,
        data=post_data,
        timeout=15,
    )

    if "ticket" not in post_resp.url and "ticket" not in post_resp.text:
        raise HTTPException(
            status_code=401,
            detail="Identifiants Garmin incorrects ou connexion refusée."
        )

    # 3. Extraire le ticket et finaliser la session sur connect.garmin.com
    ticket = None
    if "ticket=" in post_resp.url:
        ticket = post_resp.url.split("ticket=")[-1].split("&")[0]
    elif "ticket=" in post_resp.text:
        import re
        match = re.search(r'ticket=([A-Za-z0-9\-_]+)', post_resp.text)
        if match:
            ticket = match.group(1)

    if not ticket:
        raise HTTPException(status_code=401, detail="Ticket SSO Garmin introuvable.")

    # 4. Valider le ticket sur connect.garmin.com
    connect_resp = session.get(
        "https://connect.garmin.com/modern/",
        params={"ticket": ticket},
        timeout=15,
    )
    if not connect_resp.ok:
        raise HTTPException(status_code=502, detail="Impossible de valider la session Garmin Connect.")

    return session


@app.post("/garmin/upload")
async def garmin_upload(
    email: str = Form(...),
    password: str = Form(...),
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload un fichier TCX vers Garmin Connect.
    Corps multipart/form-data :
      - email    : email du compte Garmin Connect
      - password : mot de passe Garmin Connect
      - file     : fichier .tcx
    """
    # Lire le contenu du fichier
    tcx_content = await file.read()
    if not tcx_content:
        raise HTTPException(status_code=400, detail="Fichier TCX vide.")

    # Authentification Garmin
    try:
        session = garmin_login(email, password)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur connexion Garmin : {str(e)}")

    # Upload du fichier TCX
    try:
        upload_resp = session.post(
            GARMIN_UPLOAD_URL,
            files={"data": (file.filename or "seance.tcx", tcx_content, "application/vnd.garmin.tcx+xml")},
            headers={
                "NK": "NT",
                "X-HTTP-Method-Override": "POST",
            },
            timeout=30,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur upload Garmin : {str(e)}")

    if upload_resp.status_code in (200, 201):
        try:
            result = upload_resp.json()
        except Exception:
            result = {}
        return {"status": "success", "message": "Séance uploadée vers Garmin Connect ✓", "detail": result}

    elif upload_resp.status_code == 409:
        return {"status": "duplicate", "message": "Cette séance existe déjà dans Garmin Connect."}

    else:
        raise HTTPException(
            status_code=upload_resp.status_code,
            detail=f"Garmin Connect a refusé l'upload : {upload_resp.text[:300]}"
        )
