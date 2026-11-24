
import pytest
from datetime import datetime, timedelta, timezone

from backend.app.services import draft_service
from backend.app.services.draft_service import (
    DraftService,
    IneligibleLeadError,
    DraftNotFoundError,
)


# --------------------------------------------------
# Mock Gemini client
# --------------------------------------------------

class MockGemini:

    def __init__(self, responses=None):
        self.responses = list(responses) if responses is not None else []
        self.call_count = 0
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        self.call_count += 1

        if self.responses:
            return self.responses.pop(0)

        return (
            {
                "subject": "Python Developer Opportunity",
                "body": (
                    "We found a Python Developer opportunity "
                    "that may interest you."
                ),
                "claims_used": [
                    {"fact_id": "fact_role"},
                ],
            },
            {"total_tokens": 100},
        )


# --------------------------------------------------
# Test fixtures
# --------------------------------------------------

@pytest.fixture
def opportunity():
    return {
        "id": "opp-123",
        "title": "Python Developer",
        "location": "Bengaluru",
        "source_id": "official_careers",
        "source_name": "Official Careers",
        "source_status": "healthy",
        "source_is_enabled": True,
        "description": "Python developer role",
        "salary_text": "12 LPA",
        "published_at": datetime.now(timezone.utc),
        "is_suppressed": False,
    }


@pytest.fixture
def campaign():
    return {
        "id": "campaign-123",
        "name": "Developer Outreach",
    }


@pytest.fixture
def contact():
    return {
        "id": "contact-123",
        "role_title": "HR Manager",
        "verified_status": "deliverable",
    }


@pytest.fixture
def prompt_files(tmp_path, monkeypatch):
    base_prompt = tmp_path / "email_draft_v1.md"
    tone_prompt = tmp_path / "tone_hr.md"

    base_prompt.write_text(
        "Generate an email using only verified facts.",
        encoding="utf-8",
    )

    tone_prompt.write_text(
        "Use a professional HR tone.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        draft_service,
        "PROMPT_PATH",
        base_prompt,
    )

    monkeypatch.setattr(
        draft_service,
        "TONE_PROMPT_PATHS",
        {"hr": tone_prompt},
    )


@pytest.fixture
def service(
    monkeypatch,
    opportunity,
    campaign,
    contact,
    prompt_files,
):
    monkeypatch.setattr(
        draft_service,
        "is_eligible_for_campaign",
        lambda opportunity: "Eligible",
    )

    mock_gemini = MockGemini()

    instance = DraftService(
        db=object(),
        gemini_client=mock_gemini,
    )

    # Mock opportunity database lookup.
    monkeypatch.setattr(
        instance,
        "_get_opportunity",
        lambda opportunity_id: opportunity,
    )

    # Mock campaign and contact database lookup.
    # Parameter names must match the service method.
    monkeypatch.setattr(
        instance,
        "_get_campaign_and_contact",
        lambda campaign_id, opportunity_id,
        recipient_contact_method_id: (campaign, contact),
    )

    # Mock database save.
    saved_draft = {}

    def fake_save_draft(**kwargs):
        saved_draft.update(kwargs)

        return {
            "draft_id": "draft-123",
            "campaign_id": kwargs["campaign_id"],
            "status": kwargs["draft_status"],
            "ai_output": kwargs["ai_output"],
        }

    monkeypatch.setattr(
        instance,
        "_save_draft",
        fake_save_draft,
    )

    instance.saved_draft = saved_draft

    return instance


# --------------------------------------------------
# Test successful draft generation
# --------------------------------------------------

def test_generate_draft_calls_mock_gemini_client(service):

    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opp-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    assert service.gemini_client.call_count == 1
    assert len(service.gemini_client.prompts) == 1

    assert "fact_role" in service.gemini_client.prompts[0]

    assert result["status"] == "generated"
    assert result["draft_id"] == "draft-123"

    assert result["ai_output"]["subject"]
    assert result["ai_output"]["body"]

    assert service.saved_draft["draft_status"] == "generated"
    assert service.saved_draft["ai_output"] is not None


# --------------------------------------------------
# Test score below 80
# --------------------------------------------------

def test_generate_draft_rejects_score_below_80(
    service,
    monkeypatch,
    campaign,
):
    low_score_opportunity = {
        "id": "opp-low",
        "title": "Developer",
        "source_id": "aggregator",
        "source_name": "Aggregator",
        "source_status": "healthy",
        "source_is_enabled": True,
        "description": None,
        "salary_text": None,
        "location": None,
        "published_at": (
            datetime.now(timezone.utc) - timedelta(days=30)
        ),
        "is_suppressed": False,
    }

    low_score_contact = {
        "id": "contact-low",
        "role_title": None,
        "verified_status": "unknown",
    }

    score_result = draft_service.calculate_data_quality_score(
        source_type="aggregator",
        has_contact_role=False,
        verification_status="unknown",
        published_days_ago=30,
        has_description=False,
        has_salary=False,
        has_location=False,
        is_suppressed=False,
    )

    assert score_result["score"] < 80
    assert score_result["is_eligible"] is False

    monkeypatch.setattr(
        service,
        "_get_opportunity",
        lambda opportunity_id: low_score_opportunity,
    )

    monkeypatch.setattr(
        service,
        "_get_campaign_and_contact",
        lambda campaign_id, opportunity_id,
        recipient_contact_method_id: (
            campaign,
            low_score_contact,
        ),
    )

    with pytest.raises(IneligibleLeadError):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-low",
            recipient_contact_method_id="contact-low",
            tone="hr",
        )

    assert service.gemini_client.call_count == 0


# --------------------------------------------------
# Test suppressed lead
# --------------------------------------------------
def test_generate_draft_rejects_suppressed_lead(
    service,
    monkeypatch,
):
    suppressed_opportunity = {
        "id": "opp-suppressed",
        "title": "Python Developer",
        "source_id": "official_careers",
        "source_name": "Official Careers",
        "source_status": "healthy",
        "source_is_enabled": True,
        "description": "Python developer role",
        "salary_text": "12 LPA",
        "location": "Bengaluru",
        "published_at": datetime.now(timezone.utc),
    }

    monkeypatch.setattr(
        service,
        "_get_opportunity",
        lambda opportunity_id: suppressed_opportunity,
    )

    monkeypatch.setattr(
        service,
        "_is_contact_suppressed",
        lambda contact: True,
    )

    with pytest.raises(IneligibleLeadError):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-suppressed",
            recipient_contact_method_id="contact-123",
            tone="hr",
        )

    assert service.gemini_client.call_count == 0


# --------------------------------------------------
# Test validation retry and needs_review
# --------------------------------------------------

def test_validation_retry_then_needs_review(service):

    invalid_output = {
        "subject": "Test subject",
        "body": "",
        "claims_used": [],
    }

    service.gemini_client = MockGemini(
        responses=[
            (invalid_output, {"total_tokens": 50}),
            (invalid_output, {"total_tokens": 60}),
        ]
    )

    result = service.generate_draft(
        campaign_id="campaign-123",
        opportunity_id="opp-123",
        recipient_contact_method_id="contact-123",
        tone="hr",
    )

    assert service.gemini_client.call_count == 2
    assert len(service.gemini_client.prompts) == 2

    assert result["status"] == "needs_review"

    assert result["ai_output"] is not None
    assert result["ai_output"]["review_required"] is True
    assert result["ai_output"]["validation_issues"]

    assert service.saved_draft["draft_status"] == "needs_review"
    assert service.saved_draft["ai_output"] is not None


# --------------------------------------------------
# Test invalid tone
# --------------------------------------------------

def test_invalid_tone_is_rejected(service):

    with pytest.raises(ValueError, match="Invalid tone"):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="opp-123",
            recipient_contact_method_id="contact-123",
            tone="casual",
        )

    assert service.gemini_client.call_count == 0


# --------------------------------------------------
# Test missing opportunity
# --------------------------------------------------

def test_missing_opportunity_raises_not_found(
    service,
    monkeypatch,
):

    monkeypatch.setattr(
        service,
        "_get_opportunity",
        lambda opportunity_id: None,
    )

    with pytest.raises(
        DraftNotFoundError,
        match="Opportunity not found",
    ):
        service.generate_draft(
            campaign_id="campaign-123",
            opportunity_id="missing",
            recipient_contact_method_id="contact-123",
            tone="hr",
        )
