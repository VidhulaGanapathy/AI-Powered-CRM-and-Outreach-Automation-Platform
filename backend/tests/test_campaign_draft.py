from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core import security
from backend.app.api.v1 import campaigns as campaigns_api
from backend.app.services.draft_service import IneligibleLeadError


client = TestClient(app)


def test_non_eligible_lead_returns_409(monkeypatch):
    class FakeDraftService:
        def __init__(self, **kwargs):
            pass

        def generate_draft(self, **kwargs):
            raise IneligibleLeadError(
                "This lead is not eligible for campaign outreach."
            )

    # Prevent the test from accessing PostgreSQL.
    monkeypatch.setattr(
        campaigns_api,
        "get_pool",
        lambda: object(),
    )

    # Authenticate as admin.
    app.dependency_overrides[security.get_current_user] = lambda: {
        "id": "admin-1",
        "email": "admin@example.com",
        "role": "admin",
    }

    monkeypatch.setattr(
        campaigns_api,
        "DraftService",
        FakeDraftService,
    )

    try:
        response = client.post(
            "/api/v1/campaigns/campaign-123/drafts",
            json={
                "opportunity_id": "opportunity-123",
                "recipient_contact_method_id": "contact-123",
                "tone": "hr",
            },
        )

        assert response.status_code == 409
        assert "not eligible" in response.json()["detail"].lower()

    finally:
        app.dependency_overrides.pop(
            security.get_current_user,
            None,
        )
