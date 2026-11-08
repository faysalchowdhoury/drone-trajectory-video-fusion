"""Georeferencing API.

``POST /v1/homography``  >= 4 ground-control pairs (road metres <-> pixels) -> H and reprojection RMSE
``POST /v1/georeference`` pixel points + H -> road coordinates
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..geometry import apply, dlt, reprojection_rmse


class GCPRequest(BaseModel):
    road_m: list[tuple[float, float]] = Field(min_length=4)
    pixels: list[tuple[float, float]] = Field(min_length=4)


class GeoRequest(BaseModel):
    h: list[list[float]]
    pixels: list[tuple[float, float]] = Field(min_length=1, max_length=10_000)


def create_app() -> FastAPI:
    app = FastAPI(title="Drone georeferencing", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/homography")
    def homography(r: GCPRequest) -> dict[str, object]:
        if len(r.road_m) != len(r.pixels):
            raise HTTPException(422, "road_m and pixels must have equal length")
        src, dst = np.asarray(r.road_m, float), np.asarray(r.pixels, float)
        try:
            h = dlt(src, dst)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise HTTPException(422, f"degenerate control points: {exc}") from exc
        return {"h": h.tolist(), "reprojection_rmse_px": reprojection_rmse(h, src, dst)}

    @app.post("/v1/georeference")
    def georef(r: GeoRequest) -> dict[str, object]:
        h = np.asarray(r.h, float)
        if h.shape != (3, 3) or abs(np.linalg.det(h)) < 1e-12:
            raise HTTPException(422, "h must be an invertible 3x3 matrix")
        return {"road_m": apply(np.linalg.inv(h), np.asarray(r.pixels, float)).tolist()}

    return app
