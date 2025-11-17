import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import requests

THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

app = FastAPI(title="Soccer Match Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TeamSummary(BaseModel):
    id: str
    name: str
    badge: Optional[str] = None
    country: Optional[str] = None
    league: Optional[str] = None


class LikelyScorer(BaseModel):
    name: str
    probability: float = Field(..., ge=0, le=1)
    recent_goals: int


class TeamFormStats(BaseModel):
    team: TeamSummary
    last5_points: int
    last5_record: str
    avg_goals_for: float
    avg_goals_against: float
    clean_sheets: int


class MatchAnalysis(BaseModel):
    home: TeamFormStats
    away: TeamFormStats
    expected_goals: Dict[str, float]
    win_probabilities: Dict[str, float]
    likely_scorers: Dict[str, List[LikelyScorer]]
    methodology: str
    source: str


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


# --------- Helper functions for TheSportsDB ---------

def tsdb_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{THESPORTSDB_BASE}/{path}"
    r = requests.get(url, params=params or {}, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream error: {r.status_code}")
    return r.json() or {}


def search_teams(q: str) -> List[TeamSummary]:
    data = tsdb_get("searchteams.php", {"t": q})
    teams = data.get("teams") or []
    results: List[TeamSummary] = []
    for t in teams:
        results.append(TeamSummary(
            id=str(t.get("idTeam")),
            name=t.get("strTeam"),
            badge=t.get("strTeamBadge"),
            country=t.get("strCountry"),
            league=t.get("strLeague")
        ))
    return results


def get_team_last_events(team_id: str, n: int = 5) -> List[Dict[str, Any]]:
    data = tsdb_get("eventslast.php", {"id": team_id})
    events = data.get("results") or []
    return events[:n]


def summarize_team(team_id: str) -> TeamSummary:
    data = tsdb_get("lookupteam.php", {"id": team_id})
    teams = data.get("teams") or []
    if not teams:
        raise HTTPException(status_code=404, detail="Team not found")
    t = teams[0]
    return TeamSummary(
        id=str(t.get("idTeam")),
        name=t.get("strTeam"),
        badge=t.get("strTeamBadge"),
        country=t.get("strCountry"),
        league=t.get("strLeague"),
    )


def parse_goals_from_event(event: Dict[str, Any], as_home: bool) -> int:
    # TheSportsDB stores score as int fields
    if as_home:
        try:
            return int(event.get("intHomeScore") or 0)
        except Exception:
            return 0
    else:
        try:
            return int(event.get("intAwayScore") or 0)
        except Exception:
            return 0


def parse_conceded_from_event(event: Dict[str, Any], as_home: bool) -> int:
    return parse_goals_from_event(event, not as_home)


def parse_result_points(event: Dict[str, Any], as_home: bool) -> int:
    hs = parse_goals_from_event(event, True)
    as_ = parse_goals_from_event(event, False)
    if hs == as_:
        return 1
    if as_home and hs > as_:
        return 3
    if (not as_home) and as_ > hs:
        return 3
    return 0


def extract_goal_scorers(event: Dict[str, Any], as_home: bool) -> List[str]:
    key = "strHomeGoalDetails" if as_home else "strAwayGoalDetails"
    details = event.get(key) or ""
    if not details:
        return []
    # Format example: "45': John Doe; 78': Jane Roe"
    parts = [p.strip() for p in details.split(";") if p.strip()]
    names: List[str] = []
    for p in parts:
        # Remove minute and cards markers
        if ":" in p:
            try:
                name = p.split(":", 1)[1].strip()
            except Exception:
                name = p
        else:
            name = p
        # Clean trailing annotations
        name = name.replace("(pen)", "").replace("(o.g.)", "").strip()
        if name:
            names.append(name)
    return names


def build_form_stats(team_id: str, is_home: bool) -> TeamFormStats:
    team = summarize_team(team_id)
    events = get_team_last_events(team_id, 8)
    points = 0
    gf_total = 0
    ga_total = 0
    clean = 0
    record = []  # W/D/L
    for e in events[:5]:
        p = parse_result_points(e, as_home=is_home if e.get("idHomeTeam") == team_id else not is_home)
        points += p
        gf = parse_goals_from_event(e, as_home=(e.get("idHomeTeam") == team_id))
        ga = parse_conceded_from_event(e, as_home=(e.get("idHomeTeam") == team_id))
        gf_total += gf
        ga_total += ga
        if ga == 0:
            clean += 1
        if p == 3:
            record.append("W")
        elif p == 1:
            record.append("D")
        else:
            record.append("L")
    games = max(1, min(5, len(events)))
    return TeamFormStats(
        team=team,
        last5_points=points,
        last5_record="".join(record[:5]),
        avg_goals_for=round(gf_total / games, 2),
        avg_goals_against=round(ga_total / games, 2),
        clean_sheets=clean,
    )


def poisson_probabilities(lambda_home: float, lambda_away: float) -> Dict[str, float]:
    # Compute probabilities of home win/draw/away under independent Poisson goals
    from math import exp
    max_goals = 10
    probs = {"home": 0.0, "draw": 0.0, "away": 0.0}
    # Precompute Poisson pmf
    fact = [1]
    for i in range(1, max_goals + 1):
        fact.append(fact[-1] * i)

    def pois(k, lam):
        return (lam ** k) * exp(-lam) / fact[k]

    p_home = [pois(i, lambda_home) for i in range(max_goals + 1)]
    p_away = [pois(i, lambda_away) for i in range(max_goals + 1)]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = p_home[i] * p_away[j]
            if i > j:
                probs["home"] += p
            elif i == j:
                probs["draw"] += p
            else:
                probs["away"] += p
    # Normalize minor truncation loss
    s = probs["home"] + probs["draw"] + probs["away"]
    for k in probs:
        probs[k] = round(probs[k] / s, 4)
    return probs


def estimate_expected_goals(home_stats: TeamFormStats, away_stats: TeamFormStats) -> Dict[str, float]:
    # Baselines from typical league averages
    base_home = 1.45
    base_away = 1.15
    # Attack strength ~ goals for vs global avg; Defense strength ~ goals against vs global avg
    home_attack = max(0.5, min(1.5, home_stats.avg_goals_for / 1.3))
    home_defense = max(0.5, min(1.5, (1.0 / max(0.3, home_stats.avg_goals_against)) / (1.0 / 1.0)))
    away_attack = max(0.5, min(1.5, away_stats.avg_goals_for / 1.2))
    away_defense = max(0.5, min(1.5, (1.0 / max(0.3, away_stats.avg_goals_against)) / (1.0 / 1.0)))

    lambda_home = base_home * home_attack * away_defense
    lambda_away = base_away * away_attack * home_defense

    return {
        "home": round(lambda_home, 2),
        "away": round(lambda_away, 2)
    }


def compute_likely_scorers(home_id: str, away_id: str) -> Dict[str, List[LikelyScorer]]:
    def scorer_list(team_id: str, as_home: bool) -> List[LikelyScorer]:
        events = get_team_last_events(team_id, 8)
        goals: Dict[str, int] = {}
        total = 0
        for e in events:
            as_team_home = (e.get("idHomeTeam") == team_id)
            names = extract_goal_scorers(e, as_home=as_team_home)
            for n in names:
                goals[n] = goals.get(n, 0) + 1
                total += 1
        if total == 0:
            return []
        top = sorted(goals.items(), key=lambda x: x[1], reverse=True)[:5]
        return [LikelyScorer(name=k, probability=round(v / total, 3), recent_goals=v) for k, v in top]

    return {
        "home": scorer_list(home_id, True),
        "away": scorer_list(away_id, False)
    }


@app.get("/api/search/teams", response_model=List[TeamSummary])
def api_search_teams(q: str = Query(..., min_length=2, description="Team name search")):
    return search_teams(q)


@app.get("/api/analyze", response_model=MatchAnalysis)
def api_analyze(home_id: str = Query(..., description="Home team ID from TheSportsDB"),
                away_id: str = Query(..., description="Away team ID from TheSportsDB")):
    try:
        home_stats = build_form_stats(home_id, True)
        away_stats = build_form_stats(away_id, False)
        xg = estimate_expected_goals(home_stats, away_stats)
        probs = poisson_probabilities(xg["home"], xg["away"])
        scorers = compute_likely_scorers(home_id, away_id)
        methodology = (
            "Form-based Poisson model using last 5 results from TheSportsDB. "
            "Expected goals estimated from recent goals for/against with league-average baselines. "
            "Win probabilities derived from independent Poisson goal distributions. "
            "Likely scorers inferred from recent goal contributors."
        )
        return MatchAnalysis(
            home=home_stats,
            away=away_stats,
            expected_goals=xg,
            win_probabilities=probs,
            likely_scorers=scorers,
            methodology=methodology,
            source="TheSportsDB (public demo key)"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
