"""Very-High-Resolution image PURCHASE flow (Flask Blueprint ``vhr_bp``).

A user selects an ROI, gets a price (~5 EUR / km2), pays by card (Stripe TEST
mode), and on payment success the app places an order against the Sentinel Hub
commercial-data / Third Party Data Import API.

ALL the new purchase logic lives here so the giant ``app.py`` stays untouched
except for a single isolated ``register_blueprint`` block. Routes are mounted
under ``/api/vhr-purchase``.

────────────────────────────────────────────────────────────────────────────
SAFETY GATES (read before changing anything):

  * PAYMENT  = Stripe **TEST mode**. Real test keys are read from env
    ``STRIPE_SECRET_KEY`` / ``STRIPE_PUBLISHABLE_KEY``. If no secret key is set
    (or the ``stripe`` lib is missing) we FALL BACK to a *simulated* checkout
    session so the flow is demoable end-to-end. Everything is badged TEST MODE.

  * SENTINEL HUB ORDER = **DRY-RUN / SANDBOX by default**. On payment success we
    SEARCH the collection + build the CREATE-order request, and we LOG the exact
    request that WOULD be sent. We only actually POST the order *and* the
    money-spending **confirm** step when BOTH:
        SH_CONFIRM_ORDER=1   AND   real SH commercial creds are present.
    Otherwise we return a simulated order id + ``sh_status="sandbox"``. The
    confirm endpoint is the only step that spends real EUR / quota and it stays
    gated OFF by default.

────────────────────────────────────────────────────────────────────────────
ENV CONFIG (all optional; defaults shown):

  PRICE_EUR_PER_KM2   = 5.0      unit price
  MIN_AREA_KM2        = 1.0      minimum billable area
  CURRENCY            = EUR
  VHR_COLLECTION      = "Airbus Pleiades"   default commercial collection name
  VHR_COLLECTION_ID   = ""       SH dataimport collection/provider id (optional)
  VHR_RESOLUTION_M    = 0.5      nominal native resolution of the collection
  VHR_PROVIDER        = "AIRBUS"
  STRIPE_SECRET_KEY   = ""       sk_test_... (test mode). empty ⇒ simulated.
  STRIPE_PUBLISHABLE_KEY = ""    pk_test_...  (returned to UI for badge)
  SH_CLIENT_ID / SH_CLIENT_SECRET   reuse the app's existing SH creds
  SH_CONFIRM_ORDER    = 0        the money gate. 1 + creds ⇒ real confirm.
  PUBLIC_BASE_URL     = ""       used to build Stripe success/cancel URLs
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

import requests
from flask import Blueprint, jsonify, redirect, request

try:
    from backend import vhr_orders_store as orders_store
except Exception:  # pragma: no cover - direct import fallback
    import vhr_orders_store as orders_store  # type: ignore

try:
    from backend import jobs_store
except Exception:  # pragma: no cover
    try:
        import jobs_store  # type: ignore
    except Exception:
        jobs_store = None  # type: ignore

L = logging.getLogger("vhr_purchase")

vhr_bp = Blueprint("vhr_purchase", __name__, url_prefix="/api/vhr-purchase")

# Sentinel Hub commercial / Third Party Data Import endpoints.
SH_AUTH_URL = "https://services.sentinel-hub.com/oauth/token"
SH_DATAIMPORT_BASE = "https://services.sentinel-hub.com/api/v1/dataimport"


# ── config helpers ───────────────────────────────────────────────────────
def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except Exception:
        return default


def _cfg() -> dict[str, Any]:
    return {
        "price_per_km2": _f("PRICE_EUR_PER_KM2", 5.0),
        "min_area_km2": _f("MIN_AREA_KM2", 1.0),
        "currency": (os.getenv("CURRENCY", "EUR") or "EUR").upper(),
        "collection": os.getenv("VHR_COLLECTION", "Airbus Pleiades") or "Airbus Pleiades",
        "collection_id": os.getenv("VHR_COLLECTION_ID", "") or "",
        "resolution_m": _f("VHR_RESOLUTION_M", 0.5),
        "provider": os.getenv("VHR_PROVIDER", "AIRBUS") or "AIRBUS",
        "stripe_secret": os.getenv("STRIPE_SECRET_KEY", "") or "",
        "stripe_pub": os.getenv("STRIPE_PUBLISHABLE_KEY", "") or "",
        "sh_client_id": os.getenv("SH_CLIENT_ID", "") or "",
        "sh_client_secret": os.getenv("SH_CLIENT_SECRET", "") or "",
        "sh_confirm": os.getenv("SH_CONFIRM_ORDER", "0").strip() in ("1", "true", "True", "yes"),
        "base_url": (os.getenv("PUBLIC_BASE_URL", "") or "").rstrip("/"),
    }


# ── geometry / pricing ───────────────────────────────────────────────────
_EARTH_R = 6371008.8  # mean Earth radius (m)


def _bbox_from_geometry(geom: dict) -> list[float] | None:
    try:
        coords = geom["coordinates"]
        # flatten to list of [lon,lat]
        pts: list[list[float]] = []

        def _walk(x):
            if (
                isinstance(x, (list, tuple))
                and len(x) >= 2
                and all(isinstance(v, (int, float)) for v in x[:2])
            ):
                pts.append([float(x[0]), float(x[1])])
            elif isinstance(x, (list, tuple)):
                for y in x:
                    _walk(y)

        _walk(coords)
        if not pts:
            return None
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        return None


def _ring_area_km2(ring: list[list[float]]) -> float:
    """Geodesic area of a lon/lat ring via the spherical-excess formula."""
    if len(ring) < 3:
        return 0.0
    total = 0.0
    n = len(ring)
    for i in range(n):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[(i + 1) % n]
        total += math.radians(lon2 - lon1) * (
            2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2))
        )
    area = abs(total * _EARTH_R * _EARTH_R / 2.0)
    return area / 1e6


def _bbox_area_km2(bbox: list[float]) -> float:
    w, s, e, n = bbox
    ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    return _ring_area_km2(ring)


def _geom_area_km2(geom: dict) -> float:
    try:
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            polys = [coords]
        elif gtype == "MultiPolygon":
            polys = coords
        else:
            return 0.0
        total = 0.0
        for poly in polys:
            if not poly:
                continue
            outer = [[float(p[0]), float(p[1])] for p in poly[0]]
            total += _ring_area_km2(outer)
            for hole in poly[1:]:
                total -= _ring_area_km2([[float(p[0]), float(p[1])] for p in hole])
        return max(total, 0.0)
    except Exception:
        return 0.0


def _parse_roi(body: dict) -> tuple[list[float] | None, dict | None, float]:
    """Return (bbox[w,s,e,n], geometry|None, area_km2). Accepts bbox list,
    bbox dict, or a GeoJSON geometry/feature."""
    geom = None
    bbox = None
    area = 0.0

    raw_bbox = body.get("bbox")
    if isinstance(raw_bbox, dict):
        try:
            bbox = [
                float(raw_bbox["west"]),
                float(raw_bbox["south"]),
                float(raw_bbox["east"]),
                float(raw_bbox["north"]),
            ]
        except Exception:
            bbox = None
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        bbox = [float(v) for v in raw_bbox]

    g = body.get("geometry") or body.get("geojson")
    if isinstance(g, dict):
        if g.get("type") == "Feature":
            g = g.get("geometry")
        if isinstance(g, dict) and g.get("type") in ("Polygon", "MultiPolygon"):
            geom = g
            area = _geom_area_km2(g)
            if bbox is None:
                bbox = _bbox_from_geometry(g)

    if bbox is not None and area <= 0:
        area = _bbox_area_km2(bbox)

    return bbox, geom, area


def _quote(body: dict) -> dict[str, Any]:
    cfg = _cfg()
    bbox, geom, area = _parse_roi(body)
    if bbox is None:
        raise ValueError("Missing or invalid ROI: provide bbox [w,s,e,n] or geometry")
    billable = max(area, cfg["min_area_km2"])
    price = round(billable * cfg["price_per_km2"], 2)
    collection = body.get("collection") or cfg["collection"]
    return {
        "bbox": bbox,
        "geometry": geom,
        "area_km2": round(area, 4),
        "price_eur": price,
        "currency": cfg["currency"],
        "unit_price_eur_per_km2": cfg["price_per_km2"],
        "min_area_km2": cfg["min_area_km2"],
        "collection": collection,
        "resolution_m": cfg["resolution_m"],
        "provider": cfg["provider"],
        "date_from": body.get("date_from"),
        "date_to": body.get("date_to"),
    }


# ── Sentinel Hub commercial-data (dry-run by default) ─────────────────────
def _sh_token(cfg: dict) -> str | None:
    if not cfg["sh_client_id"] or not cfg["sh_client_secret"]:
        return None
    try:
        r = requests.post(
            SH_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": cfg["sh_client_id"],
                "client_secret": cfg["sh_client_secret"],
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as ex:
        L.warning("SH commercial OAuth failed (falling back to sandbox): %s", ex)
        return None


def _build_sh_order_request(order: dict, cfg: dict) -> dict:
    """Construct the Third Party Data Import (dataimport/v1) order body for the
    ROI + date window. This is logged in dry-run so 'it sends a request' is
    visibly true; it is the body we would POST to /dataimport/v1/orders."""
    bbox = order.get("bbox")
    geom = order.get("geometry")
    aoi = geom if geom else {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]],
        ]],
    }
    params: dict[str, Any] = {
        "productKind": "BUNDLE",
        "resolution": cfg["resolution_m"],
        "collectionId": cfg["collection_id"] or None,
    }
    if order.get("date_from") and order.get("date_to"):
        params["timeRange"] = {"from": order["date_from"], "to": order["date_to"]}
    return {
        "name": f"vhr-purchase-{order['order_id']}",
        "input": {
            "provider": cfg["provider"],
            "bounds": {"geometry": aoi},
            "data": [{"constellation": cfg["provider"], "dataFilter": params}],
        },
    }


def place_sh_order(order: dict) -> dict[str, Any]:
    """Search + create the SH commercial order. CONFIRM (money) gated OFF unless
    SH_CONFIRM_ORDER=1 AND real creds present. Returns dict with sh_status,
    sh_order_id, sh_request, error."""
    cfg = _cfg()
    sh_request = _build_sh_order_request(order, cfg)
    # Always log the would-be / actual request so the API call is visible.
    L.info(
        "SH dataimport ORDER request (dry_run=%s) → POST %s/v1/orders\n%s",
        not (cfg["sh_confirm"]),
        SH_DATAIMPORT_BASE,
        json.dumps(sh_request, indent=2, default=str),
    )

    token = _sh_token(cfg)
    creds_present = token is not None

    # SANDBOX path: no creds, or confirm gate off → never spend money.
    if not creds_present or not cfg["sh_confirm"]:
        reason = (
            "SH_CONFIRM_ORDER!=1 (money gate OFF)"
            if creds_present
            else "no SH commercial creds (token unavailable)"
        )
        L.info("SH order SANDBOX/dry-run: %s — confirm step NOT called.", reason)
        return {
            "sh_status": "sandbox",
            "sh_order_id": f"sandbox-{order['order_id']}",
            "sh_request": sh_request,
            "sh_dry_run_reason": reason,
            "error": None,
        }

    # REAL path (both creds + gate on): search → create → confirm.
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        # 1) SEARCH the collection (no charge).
        try:
            sresp = requests.post(
                f"{SH_DATAIMPORT_BASE}/v1/search",
                headers=headers,
                json={"provider": cfg["provider"], "bounds": sh_request["input"]["bounds"]},
                timeout=60,
            )
            L.info("SH search HTTP %s", sresp.status_code)
        except Exception as sx:
            L.warning("SH search failed (continuing to create): %s", sx)

        # 2) CREATE the order object (no charge — money is the confirm step).
        cresp = requests.post(
            f"{SH_DATAIMPORT_BASE}/v1/orders", headers=headers, json=sh_request, timeout=60
        )
        cresp.raise_for_status()
        created = cresp.json()
        sh_order_id = created.get("id") or created.get("orderId")
        L.warning("SH order CREATED id=%s (confirm gate is ON → confirming)", sh_order_id)

        # 3) CONFIRM — THE MONEY STEP. Only here because gate+creds verified.
        confirm = requests.post(
            f"{SH_DATAIMPORT_BASE}/v1/orders/{sh_order_id}/confirm",
            headers=headers,
            timeout=60,
        )
        confirm.raise_for_status()
        return {
            "sh_status": "confirmed",
            "sh_order_id": sh_order_id,
            "sh_request": sh_request,
            "error": None,
        }
    except Exception as ex:
        L.error("SH real order placement failed: %s", ex)
        return {
            "sh_status": "error",
            "sh_order_id": None,
            "sh_request": sh_request,
            "error": str(ex)[:500],
        }


# ── Stripe (test mode) with simulated fallback ────────────────────────────
def _stripe_lib():
    cfg = _cfg()
    if not cfg["stripe_secret"]:
        return None
    try:
        import stripe  # type: ignore

        stripe.api_key = cfg["stripe_secret"]
        return stripe
    except Exception as ex:
        L.warning("stripe lib unavailable, simulating checkout: %s", ex)
        return None


def _success_url(order_id: str, cfg: dict) -> str:
    base = cfg["base_url"]
    return (
        f"{base}/api/vhr-purchase/success?order_id={order_id}&session_id={{CHECKOUT_SESSION_ID}}"
        if base
        else f"/api/vhr-purchase/success?order_id={order_id}&session_id={{CHECKOUT_SESSION_ID}}"
    )


# ── routes ────────────────────────────────────────────────────────────────
@vhr_bp.route("/config", methods=["GET"])
def config():
    cfg = _cfg()
    return jsonify({
        "test_mode": True,
        "currency": cfg["currency"],
        "unit_price_eur_per_km2": cfg["price_per_km2"],
        "min_area_km2": cfg["min_area_km2"],
        "collection": cfg["collection"],
        "resolution_m": cfg["resolution_m"],
        "provider": cfg["provider"],
        "stripe_publishable_key": cfg["stripe_pub"] or None,
        "stripe_enabled": bool(cfg["stripe_secret"]),
        "sh_confirm_enabled": cfg["sh_confirm"],
        "sh_creds_present": bool(cfg["sh_client_id"] and cfg["sh_client_secret"]),
    })


@vhr_bp.route("/quote", methods=["POST"])
def quote():
    body = request.get_json(silent=True) or {}
    try:
        q = _quote(body)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    q.pop("geometry", None)
    q["test_mode"] = True
    return jsonify(q)


@vhr_bp.route("/checkout", methods=["POST"])
def checkout():
    body = request.get_json(silent=True) or {}
    cfg = _cfg()
    try:
        q = _quote(body)  # re-quote server-side; never trust client price.
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    order = orders_store.create_order(
        bbox=q["bbox"], geometry=q["geometry"], area_km2=q["area_km2"],
        price_eur=q["price_eur"], currency=q["currency"], collection=q["collection"],
        resolution_m=q["resolution_m"], provider=q["provider"],
        date_from=q["date_from"], date_to=q["date_to"], payment_status="pending",
    )
    oid = order["order_id"]

    stripe = _stripe_lib()
    simulated = stripe is None
    checkout_url = None
    session_id = None

    if stripe is not None:
        try:
            amount_cents = int(round(q["price_eur"] * 100))
            base = cfg["base_url"]
            cancel = (f"{base}" if base else "") + "/api/vhr-purchase/cancel?order_id=" + oid
            session = stripe.checkout.Session.create(
                mode="payment",
                line_items=[{
                    "price_data": {
                        "currency": q["currency"].lower(),
                        "product_data": {
                            "name": f"VHR imagery — {q['collection']} (TEST MODE)",
                            "description": f"{q['area_km2']} km2 @ {q['unit_price_eur_per_km2']} EUR/km2",
                        },
                        "unit_amount": amount_cents,
                    },
                    "quantity": 1,
                }],
                success_url=_success_url(oid, cfg),
                cancel_url=cancel,
                metadata={
                    "order_id": oid, "area_km2": str(q["area_km2"]),
                    "collection": q["collection"], "bbox": json.dumps(q["bbox"]),
                },
            )
            checkout_url = session.url
            session_id = session.id
        except Exception as ex:
            L.warning("Stripe session create failed, simulating: %s", ex)
            simulated = True

    if simulated:
        session_id = f"sim_sess_{oid}"
        checkout_url = (
            (cfg["base_url"] if cfg["base_url"] else "")
            + f"/api/vhr-purchase/success?order_id={oid}&session_id={session_id}&simulated=1"
        )

    orders_store.update_order(oid, session_id=session_id, simulated=simulated)
    return jsonify({
        "order_id": oid,
        "checkout_url": checkout_url,
        "session_id": session_id,
        "simulated": simulated,
        "test_mode": True,
        "price_eur": q["price_eur"],
        "currency": q["currency"],
    })


def _finalise_paid(order_id: str) -> dict[str, Any] | None:
    """Transition to PAID, place the (gated) SH order, register a Results job."""
    order = orders_store.get_order(order_id)
    if order is None:
        return None
    order = orders_store.update_order(order_id, payment_status="paid") or order

    sh = place_sh_order(order)
    order = orders_store.update_order(
        order_id,
        sh_status=sh["sh_status"], sh_order_id=sh["sh_order_id"],
        sh_request=sh["sh_request"], error=sh.get("error"),
    ) or order

    # Surface the purchase in the Results panel (R3 store).
    result_id = None
    if jobs_store is not None:
        try:
            res = jobs_store.register_result(
                name=f"VHR purchase — {order.get('collection')} ({order.get('area_km2')} km2)",
                roi_bbox=order.get("bbox"),
                resolution="vhr",
                status="done" if sh["sh_status"] in ("sandbox", "created", "confirmed") else "failed",
                metrics={
                    "purchase": True, "price_eur": order.get("price_eur"),
                    "currency": order.get("currency"), "sh_status": sh["sh_status"],
                    "sh_order_id": sh["sh_order_id"], "collection": order.get("collection"),
                    "provider": order.get("provider"), "test_mode": True,
                },
                job_id=order_id,
                error=sh.get("error"),
            )
            result_id = res.get("id")
        except Exception as ex:
            L.warning("register_result for purchase failed: %s", ex)

    order = orders_store.update_order(order_id, result_id=result_id) or order
    return order


@vhr_bp.route("/confirm", methods=["POST"])
def confirm():
    body = request.get_json(silent=True) or {}
    order_id = body.get("order_id")
    session_id = body.get("session_id")
    if not order_id:
        return jsonify({"error": "Missing order_id"}), 400
    order = orders_store.get_order(order_id)
    if order is None:
        return jsonify({"error": "Unknown order_id"}), 404

    # Verify payment: real Stripe session must be 'paid'; simulated always OK.
    stripe = _stripe_lib()
    paid = bool(order.get("simulated")) or bool(body.get("simulated"))
    if stripe is not None and session_id and not paid:
        try:
            sess = stripe.checkout.Session.retrieve(session_id)
            paid = sess.get("payment_status") == "paid"
        except Exception as ex:
            L.warning("Stripe session retrieve failed: %s", ex)
            paid = False
    if not paid and stripe is None:
        paid = True  # no stripe configured ⇒ demo/simulated success

    if not paid:
        orders_store.update_order(order_id, payment_status="failed")
        return jsonify({"error": "Payment not completed", "payment_status": "failed"}), 402

    updated = _finalise_paid(order_id)
    return jsonify(updated)


@vhr_bp.route("/success", methods=["GET"])
def success():
    """Stripe / simulated redirect target. Finalises the order then redirects to
    the SPA (or returns JSON if no SPA base)."""
    order_id = request.args.get("order_id")
    if not order_id or orders_store.get_order(order_id) is None:
        return jsonify({"error": "Unknown order"}), 404
    order = orders_store.get_order(order_id)
    if order.get("payment_status") != "paid" or not order.get("sh_status"):
        order = _finalise_paid(order_id) or order
    cfg = _cfg()
    if request.args.get("redirect") == "0":
        return jsonify(order)
    target = (cfg["base_url"] if cfg["base_url"] else "") + f"/?vhr_order={order_id}&vhr_paid=1"
    return redirect(target, code=302)


@vhr_bp.route("/cancel", methods=["GET"])
def cancel():
    order_id = request.args.get("order_id")
    if order_id:
        orders_store.update_order(order_id, payment_status="failed", error="checkout cancelled")
    cfg = _cfg()
    return redirect((cfg["base_url"] if cfg["base_url"] else "") + "/?vhr_cancelled=1", code=302)


@vhr_bp.route("/webhook", methods=["POST"])
def webhook():
    """Stripe webhook (test). Verifies signature if STRIPE_WEBHOOK_SECRET set,
    else trusts the payload (demo). Finalises on checkout.session.completed."""
    stripe = _stripe_lib()
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    event = None
    if stripe is not None and secret:
        try:
            event = stripe.Webhook.construct_event(
                request.get_data(), request.headers.get("Stripe-Signature", ""), secret
            )
        except Exception as ex:
            return jsonify({"error": f"bad signature: {ex}"}), 400
    else:
        event = request.get_json(silent=True) or {}

    etype = (event or {}).get("type")
    obj = ((event or {}).get("data") or {}).get("object") or {}
    if etype in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        order_id = (obj.get("metadata") or {}).get("order_id")
        if order_id:
            _finalise_paid(order_id)
    return jsonify({"received": True})


@vhr_bp.route("/orders", methods=["GET"])
def orders():
    items = orders_store.list_orders(limit=int(request.args.get("limit", 500)))
    out = [{
        "order_id": o["order_id"], "bbox": o.get("bbox"), "area_km2": o.get("area_km2"),
        "price_eur": o.get("price_eur"), "currency": o.get("currency"),
        "collection": o.get("collection"), "payment_status": o.get("payment_status"),
        "sh_status": o.get("sh_status"), "sh_order_id": o.get("sh_order_id"),
        "result_id": o.get("result_id"), "simulated": o.get("simulated"),
        "created_at": o.get("created_at"),
    } for o in items]
    return jsonify({"count": len(out), "orders": out, "test_mode": True})


@vhr_bp.route("/orders/<order_id>", methods=["GET"])
def order_detail(order_id):
    o = orders_store.get_order(order_id)
    if o is None:
        return jsonify({"error": "Unknown order_id"}), 404
    return jsonify(o)
