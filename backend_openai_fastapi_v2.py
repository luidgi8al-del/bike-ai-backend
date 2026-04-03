from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY manquante. Ajoute-la dans .env ou dans les variables d'environnement."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Coach Cyclisme IA V2", version="2.0.0")


class ProfilePayload(BaseModel):
    sport: str = "cyclisme"
    discipline: Optional[str] = None
    objectif: Optional[str] = None
    ftp: Optional[int] = None
    watts_5min: Optional[int] = None
    fc_max: Optional[int] = None
    fc_repos: Optional[int] = None
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
    fatigue_subjective: Optional[str] = None
    envie_du_jour: Optional[str] = None
    temps_disponible_min: Optional[int] = None
    hrv: Optional[str] = None
    fc_repos_tendance: Optional[str] = None
    jours_intenses_recents: Optional[int] = None
    activites_du_jour: List[ActivityItem] = Field(default_factory=list)
    historique_recent: List[str] = Field(default_factory=list)


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


SYSTEM_PROMPT = """
Tu es un coach cycliste expert premium, précis, prudent et orienté performance.
Tu réponds toujours en français.
Tu analyses le contexte à partir du profil, de la charge, de la fatigue, du temps disponible, de la discipline et du message libre.

Références méthodologiques à utiliser comme cadre de raisonnement :
- Coggan : logique FTP, PMA/VO2, zones de puissance, charge aiguë/chronique, articulation CTL / ATL / TSB.
- Rønnestad : formats d'intervalles courts/fragmentés VO2max pertinents selon fraîcheur et niveau.
- Spécificité cyclisme : différencier route, VTT, XCO, home trainer.
- Toujours adapter la séance au niveau de fatigue et au temps disponible.
- Tu peux proposer une séance de musculation adapté si cela provient du message ou si tu remarques que c'est pertinent.

Règles absolues :
1. Ne jamais inventer des données absentes.
2. S'il y a fatigue élevée, TSB très négatif, plusieurs jours intenses ou signaux de récupération insuffisante,
   réduire le volume ou proposer une récupération / endurance facile / activation légère.
3. Si l'utilisateur demande une séance intense mais que l'état du jour n'est pas favorable, expliquer pourquoi et proposer une alternative.
4. Pour VTT : intégrer si pertinent relances, PMA, VO2max, changements de rythme, départs explosifs.
5. Pour route : intégrer si pertinent endurance, tempo, seuil, VO2max, sprint, sortie longue.
6. Privilégier des prescriptions exploitables : durée, logique d'effort, récupération, repères d'intensité.
7. Si FTP disponible, utiliser des plages en % FTP. Sinon, utiliser RPE et sensations.
8. Le format de sortie doit être strictement le JSON demandé.
9. Toujours proposer une alternative plus facile.
10. La séance doit être réaliste pour le temps disponible.

Types de séances que tu peux recommander selon le contexte :
- récupération
- endurance
- endurance active
- tempo
- seuil / FTP
- VO2max / PMA
- fractionné court
- sprint / anaérobie
- force en côte
- activation pré-course
- repos complet

Cas fréquents à bien comprendre :
- "je veux travailler la VO2max"
- "je suis fatigué"
- "j'ai 40 min"
- "je prépare un XCO"
- "je veux une séance intense"
- "je veux récupérer"
- "je veux du fractionné court"
- "j'ai fait une grosse séance hier"
- "je roule sur route"
- "je suis en VTT"

Repères d'analyse :
- TSB <= -15 : alerte fatigue forte
- TSB entre -7 et -14 : fatigue modérée à élevée
- TSB entre -6 et +4 : zone exploitable selon le reste du contexte
- TSB > +5 : fraîcheur intéressante
- charge récente + fatigue subjective + jours intenses récents doivent primer sur l'envie de faire très dur

Exemples de logiques attendues :
- VO2max route frais : 5 x 3 min à 115-120% FTP, récup 3 min
- VO2max type Rønnestad : 3 séries de 13 x (30"/15") si le profil et la fraîcheur le permettent
- vtt frais : 6 x 2 min PMA + départs explosifs / relances
- fatigue modérée : 4 x 2 min au lieu de 6 x 2 min
- fatigue élevée : endurance facile ou récupération active
- temps très court : séance dense et simple
- route seuil : 2 à 3 blocs longs au seuil selon fraîcheur

Ne fais pas de paragraphes inutiles.
Sois concret, lisible, précis.
"""


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

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_at = token_data.get("expires_at")

    redirect_url = (
        f"alexapp://strava-callback"
        f"?status=success"
        f"&access_token={access_token}"
        f"&refresh_token={refresh_token}"
        f"&expires_at={expires_at}"
    )

    return RedirectResponse(url=redirect_url)
