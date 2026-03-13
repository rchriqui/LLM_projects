"""
Tool functions for the Camp Registration Assistant.

Interact with mock_db.json for all camp, kid, and registration operations.
"""

import json
import uuid
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "mock_db.json"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _load_db() -> dict:
    with open(DB_PATH) as f:
        return json.load(f)


def _save_db(db: dict) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _dates_overlap(start1: date, end1: date, start2: date, end2: date) -> bool:
    return start1 <= end2 and end1 >= start2


def _time_slots_overlap(slot1: str, slot2: str) -> bool:
    """Return True if two 'HH:MM-HH:MM' time slots overlap."""
    s1, e1 = slot1.split("-")
    s2, e2 = slot2.split("-")
    return s1 < e2 and e1 > s2


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_camps(name: str | None = None, status: str | None = None, age: int | None = None) -> list[dict]:
    """
    Return camp records.

    Args:
        name:   Optional substring match on camp name (case-insensitive).
        status: Optional exact match on camp status ('open', 'cancelled', …).
        age:    Optional child age — returns only camps the child is eligible for.
    """
    camps = _load_db()["camps"]
    if name:
        camps = [c for c in camps if name.lower() in c["name"].lower()]
    if status:
        camps = [c for c in camps if c["status"] == status]
    if age is not None:
        camps = [c for c in camps if c["min_age"] <= age <= c["max_age"]]
    return camps


def get_kids(name: str | None = None, kid_id: str | None = None) -> list[dict]:
    """
    Return kid records.

    Args:
        name:   Optional substring match on the child's name (case-insensitive).
        kid_id: Optional exact match on kid_id.
    """
    kids = _load_db()["kids"]
    if kid_id:
        kids = [k for k in kids if k["kid_id"] == kid_id]
    if name:
        kids = [k for k in kids if name.lower() in k["name"].lower()]
    return kids


def get_registrations(
    kid_id: str | None = None,
    camp_id: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """
    Return registration records enriched with kid_name and camp_name.

    Args:
        kid_id:  Optional filter by kid.
        camp_id: Optional filter by camp.
        status:  Optional filter by status ('pending', 'confirmed', 'cancelled', 'waitlisted').
    """
    db = _load_db()
    regs = db["registrations"]
    if kid_id:
        regs = [r for r in regs if r["kid_id"] == kid_id]
    if camp_id:
        regs = [r for r in regs if r["camp_id"] == camp_id]
    if status:
        regs = [r for r in regs if r["status"] == status]

    kids_by_id = {k["kid_id"]: k["name"] for k in db["kids"]}
    camps_by_id = {c["camp_id"]: c["name"] for c in db["camps"]}

    return [
        {
            **r,
            "kid_name": kids_by_id.get(r["kid_id"], r["kid_id"]),
            "camp_name": camps_by_id.get(r["camp_id"], r["camp_id"]),
        }
        for r in regs
    ]


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def register_kid(kid_id: str, camp_id: str) -> dict:
    """
    Register a child for a camp with full validation.

    Validates age range, duplicate registration, schedule conflicts, and capacity.
    If the camp is full, the child is waitlisted automatically.

    Args:
        kid_id:  The kid's ID (e.g. 'kid-1'). Obtain from get_kids.
        camp_id: The camp's ID (e.g. 'camp-1'). Obtain from get_camps.

    Returns the new registration record enriched with kid_name and camp_name.
    Raises ValueError with a user-friendly message on any validation failure.
    """
    db = _load_db()

    kid = next((k for k in db["kids"] if k["kid_id"] == kid_id), None)
    if not kid:
        raise ValueError(f"No child found with ID '{kid_id}'.")

    camp = next((c for c in db["camps"] if c["camp_id"] == camp_id), None)
    if not camp:
        raise ValueError(f"No camp found with ID '{camp_id}'.")

    if camp["status"] == "cancelled":
        raise ValueError(
            f"'{camp['name']}' has been cancelled and is no longer accepting registrations."
        )

    if not (camp["min_age"] <= kid["age"] <= camp["max_age"]):
        raise ValueError(
            f"{kid['name']} is {kid['age']} years old, but '{camp['name']}' is open to "
            f"children aged {camp['min_age']}–{camp['max_age']}."
        )

    # Duplicate check (ignore previously cancelled registrations)
    duplicate = next(
        (
            r
            for r in db["registrations"]
            if r["kid_id"] == kid_id
            and r["camp_id"] == camp_id
            and r["status"] != "cancelled"
        ),
        None,
    )
    if duplicate:
        raise ValueError(
            f"{kid['name']} is already registered for '{camp['name']}' "
            f"(status: {duplicate['status']})."
        )

    # Schedule conflict check
    camp_start = _parse_date(camp["start_date"])
    camp_end = _parse_date(camp["end_date"])

    active_other_camp_ids = [
        r["camp_id"]
        for r in db["registrations"]
        if r["kid_id"] == kid_id
        and r["status"] not in ("cancelled", "waitlisted")
        and r["camp_id"] != camp_id
    ]
    for other_id in active_other_camp_ids:
        other = next((c for c in db["camps"] if c["camp_id"] == other_id), None)
        if not other:
            continue
        other_start = _parse_date(other["start_date"])
        other_end = _parse_date(other["end_date"])
        if _dates_overlap(
            camp_start, camp_end, other_start, other_end
        ) and _time_slots_overlap(camp["time_slot"], other["time_slot"]):
            raise ValueError(
                f"Schedule conflict: '{camp['name']}' "
                f"({camp['start_date']} to {camp['end_date']}, {camp['time_slot']}) "
                f"overlaps with '{other['name']}' "
                f"({other['start_date']} to {other['end_date']}, {other['time_slot']}), "
                f"which {kid['name']} is already registered for."
            )

    is_full = camp["enrolled"] >= camp["capacity"]
    new_status = "waitlisted" if is_full else "pending"

    new_reg = {
        "registration_id": f"reg-{uuid.uuid4().hex[:8]}",
        "kid_id": kid_id,
        "camp_id": camp_id,
        "status": new_status,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    db["registrations"].append(new_reg)

    if not is_full:
        camp["enrolled"] += 1

    _save_db(db)

    result = {**new_reg, "kid_name": kid["name"], "camp_name": camp["name"]}
    if is_full:
        result["message"] = (
            f"'{camp['name']}' is currently full. "
            f"{kid['name']} has been added to the waitlist."
        )
    return result


def cancel_registration(registration_id: str) -> dict:
    """
    Cancel an existing registration and update camp enrollment.

    If a spot opens up, the earliest waitlisted child for the same camp is
    automatically promoted to 'pending'.

    Args:
        registration_id: The registration's ID (e.g. 'reg-1'). Obtain from get_registrations.

    Returns the cancelled registration record enriched with kid_name and camp_name.
    Raises ValueError if the registration doesn't exist or is already cancelled.
    """
    db = _load_db()

    reg = next(
        (r for r in db["registrations"] if r["registration_id"] == registration_id),
        None,
    )
    if not reg:
        raise ValueError(f"No registration found with ID '{registration_id}'.")
    if reg["status"] == "cancelled":
        raise ValueError(f"Registration '{registration_id}' is already cancelled.")

    was_enrolled = reg["status"] in ("pending", "confirmed")
    reg["status"] = "cancelled"

    camp = next((c for c in db["camps"] if c["camp_id"] == reg["camp_id"]), None)
    promoted_reg = None

    if was_enrolled and camp:
        camp["enrolled"] = max(0, camp["enrolled"] - 1)

        # Promote the earliest waitlisted child for this camp
        waitlisted = sorted(
            [
                r
                for r in db["registrations"]
                if r["camp_id"] == reg["camp_id"] and r["status"] == "waitlisted"
            ],
            key=lambda r: r["registered_at"],
        )
        if waitlisted:
            promoted_reg = waitlisted[0]
            promoted_reg["status"] = "pending"
            camp["enrolled"] += 1

    _save_db(db)

    kid = next((k for k in db["kids"] if k["kid_id"] == reg["kid_id"]), None)
    result = {
        **reg,
        "kid_name": kid["name"] if kid else reg["kid_id"],
        "camp_name": camp["name"] if camp else reg["camp_id"],
    }
    if promoted_reg:
        promoted_kid = next(
            (k for k in db["kids"] if k["kid_id"] == promoted_reg["kid_id"]), None
        )
        result["waitlist_promoted"] = {
            "registration_id": promoted_reg["registration_id"],
            "kid_name": promoted_kid["name"]
            if promoted_kid
            else promoted_reg["kid_id"],
        }
    return result


def update_registration_status(registration_id: str, new_status: str) -> dict:
    """
    Update the status of a registration.

    Prefer cancel_registration when the intent is to cancel — it provides
    more descriptive output. Use this tool for other transitions such as
    pending → confirmed or waitlisted → pending.

    Keeps camp enrollment counts consistent and auto-promotes the earliest
    waitlisted child when a spot opens up.

    Args:
        registration_id: The registration's ID (e.g. 'reg-1'). Obtain from get_registrations.
        new_status:      Target status — one of 'pending', 'confirmed', 'cancelled', 'waitlisted'.

    Returns the updated registration record enriched with kid_name and camp_name.
    Raises ValueError for unknown registration IDs or invalid status values.
    """
    valid_statuses = {"pending", "confirmed", "cancelled", "waitlisted"}
    if new_status not in valid_statuses:
        raise ValueError(
            f"Invalid status '{new_status}'. Valid options: {', '.join(sorted(valid_statuses))}."
        )

    db = _load_db()

    reg = next(
        (r for r in db["registrations"] if r["registration_id"] == registration_id),
        None,
    )
    if not reg:
        raise ValueError(f"No registration found with ID '{registration_id}'.")

    old_status = reg["status"]
    if old_status == new_status:
        raise ValueError(
            f"Registration '{registration_id}' already has status '{new_status}'."
        )

    enrolled_statuses = {"pending", "confirmed"}
    camp = next((c for c in db["camps"] if c["camp_id"] == reg["camp_id"]), None)
    entering_enrolled = old_status not in enrolled_statuses and new_status in enrolled_statuses
    leaving_enrolled = old_status in enrolled_statuses and new_status not in enrolled_statuses

    if camp:
        if entering_enrolled:
            if camp["enrolled"] >= camp["capacity"]:
                raise ValueError(
                    f"Cannot activate registration: '{camp['name']}' is already at full capacity "
                    f"({camp['capacity']} spots). Cancel another registration first or keep this one waitlisted."
                )
            camp["enrolled"] += 1
        elif leaving_enrolled:
            camp["enrolled"] = max(0, camp["enrolled"] - 1)

    reg["status"] = new_status

    # When a spot opens up, promote the earliest waitlisted child.
    promoted_reg = None
    if leaving_enrolled and camp:
        waitlisted = sorted(
            [
                r
                for r in db["registrations"]
                if r["camp_id"] == reg["camp_id"]
                and r["status"] == "waitlisted"
                and r["registration_id"] != registration_id
            ],
            key=lambda r: r["registered_at"],
        )
        if waitlisted:
            promoted_reg = waitlisted[0]
            promoted_reg["status"] = "pending"
            camp["enrolled"] += 1

    _save_db(db)

    kid = next((k for k in db["kids"] if k["kid_id"] == reg["kid_id"]), None)
    result = {
        **reg,
        "kid_name": kid["name"] if kid else reg["kid_id"],
        "camp_name": camp["name"] if camp else reg["camp_id"],
    }
    if promoted_reg:
        promoted_kid = next(
            (k for k in db["kids"] if k["kid_id"] == promoted_reg["kid_id"]), None
        )
        result["waitlist_promoted"] = {
            "registration_id": promoted_reg["registration_id"],
            "kid_name": promoted_kid["name"] if promoted_kid else promoted_reg["kid_id"],
        }
    return result
