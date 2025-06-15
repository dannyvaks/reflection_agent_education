import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Import the FastAPI app and Pydantic models from your API file
from api import app, GradeAssessmentRequest, StudentAnswerItem, GradeAssessmentResponse, GradedAnswerItem
from reflexion_teaching_agent import ReflexionTeachingAgent


# Mock GOOGLE_API_KEY for tests, as API initialization might depend on it
@pytest.fixture(autouse=True)
def mock_env_vars_for_api(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_for_api")
    # Mock dataset path to prevent actual file loading during agent init in API
    monkeypatch.setenv("INSTRUCTION_DATASET_PATH", "mock_api_dataset.csv")
    # Prevent actual file loading by ReflexionTeachingAgent if it's initialized globally
    monkeypatch.setattr("os.path.exists", lambda x: False)


@pytest.fixture(scope="function") # Changed scope from "module" to "function"
def client():
    """Provides a TestClient instance for API testing."""
    # Ensure the global agent in api.py is either mocked or uses mocked services
    with patch('api.ReflexionTeachingAgent') as MockAgentClass:
        mock_agent_instance = MagicMock(spec=ReflexionTeachingAgent)
        MockAgentClass.return_value = mock_agent_instance

        # If 'agent' is directly imported and used in api.py, we need to patch that specific instance
        # This is a common pattern if 'agent = ReflexionTeachingAgent()' is at the global scope in api.py
        with patch('api.agent', new=mock_agent_instance):
            with TestClient(app) as test_client:
                yield test_client


def test_grade_assessment_endpoint_success(client: TestClient):
    # Retrieve the mocked agent instance from the client's app state or the patch
    # This assumes the 'api.agent' patch in the client fixture is effective.
    # Accessing the mock directly via the patch context or re-importing is safer
    # if client.app.state is not reliably populated in all test runner versions/configs.
    # For this test, we'll rely on the fixture's direct patching of 'api.agent'.
    from api import agent as mock_agent # This will be the MagicMock from the fixture

    assert isinstance(mock_agent, MagicMock), "Agent was not mocked correctly in fixture"

    # Reset mock for this specific test to ensure clean state for side_effect and call_count
    mock_agent.reset_mock()

    # Configure the mock agent's grade_student_answer method
    mock_agent.grade_student_answer.side_effect = [
        {"score": 9.0, "feedback": "Excellent!"},
        {"score": 7.5, "feedback": "Good, but needs more detail."},
    ]

    payload = GradeAssessmentRequest(
        answers=[
            StudentAnswerItem(question="Q1", student_answer="SA1", correct_answer="CA1"),
            StudentAnswerItem(question="Q2", student_answer="SA2", correct_answer="CA2"),
        ]
    )

    response = client.post("/grade-assessment/", json=payload.model_dump())

    assert response.status_code == 200
    response_data = response.json()

    assert len(response_data["graded_answers"]) == 2
    assert response_data["total_score"] == 16.5  # 9.0 + 7.5

    assert response_data["graded_answers"][0]["question"] == "Q1"
    assert response_data["graded_answers"][0]["score"] == 9.0
    assert response_data["graded_answers"][0]["feedback"] == "Excellent!"

    assert response_data["graded_answers"][1]["question"] == "Q2"
    assert response_data["graded_answers"][1]["score"] == 7.5
    assert response_data["graded_answers"][1]["feedback"] == "Good, but needs more detail."

    # Verify that the mock was called correctly
    mock_agent.grade_student_answer.assert_any_call(question="Q1", student_answer="SA1", correct_answer="CA1")
    mock_agent.grade_student_answer.assert_any_call(question="Q2", student_answer="SA2", correct_answer="CA2")
    assert mock_agent.grade_student_answer.call_count == 2


def test_grade_assessment_empty_answers(client: TestClient):
    payload = GradeAssessmentRequest(answers=[])
    response = client.post("/grade-assessment/", json=payload.model_dump())

    assert response.status_code == 200
    response_data = response.json()
    assert len(response_data["graded_answers"]) == 0
    assert response_data["total_score"] == 0.0


def test_grade_assessment_item_grading_fails(client: TestClient):
    from api import agent as mock_agent
    assert isinstance(mock_agent, MagicMock)
    mock_agent.reset_mock()

    # Simulate one success and one failure
    mock_agent.grade_student_answer.side_effect = [
        {"score": 8.0, "feedback": "Correct!"},
        Exception("Grading failed for this item"), # This will be caught by the endpoint
    ]

    payload = GradeAssessmentRequest(
        answers=[
            StudentAnswerItem(question="Q1", student_answer="SA1", correct_answer="CA1"),
            StudentAnswerItem(question="Q2", student_answer="SA2", correct_answer="CA2"), # This one will "fail"
        ]
    )

    response = client.post("/grade-assessment/", json=payload.model_dump())

    assert response.status_code == 200
    response_data = response.json()

    assert len(response_data["graded_answers"]) == 2

    # First item should be graded successfully
    assert response_data["graded_answers"][0]["question"] == "Q1"
    assert response_data["graded_answers"][0]["score"] == 8.0
    assert response_data["graded_answers"][0]["feedback"] == "Correct!"

    # Second item should reflect the failure
    assert response_data["graded_answers"][1]["question"] == "Q2"
    assert response_data["graded_answers"][1]["score"] == 0.0
    assert "Failed to grade this answer: Grading failed for this item" in response_data["graded_answers"][1]["feedback"]

    # Total score should only include the successfully graded item
    assert response_data["total_score"] == 8.0
    assert mock_agent.grade_student_answer.call_count == 2


# Test for agent not initialized (harder to control global agent state from here without specific setup)
# This test assumes that the `api.agent` can be None.
# The client fixture already patches `api.agent` to be a MagicMock.
# To test the `agent is None` path, we need to modify the fixture or patch `api.agent` directly here.
def test_grade_assessment_agent_not_initialized(client: TestClient):
     # Temporarily patch api.agent to be None for this specific test
    with patch('api.agent', new=None):
        payload = GradeAssessmentRequest(
            answers=[StudentAnswerItem(question="Q1", student_answer="SA1", correct_answer="CA1")]
        )
        response = client.post("/grade-assessment/", json=payload.model_dump())
        assert response.status_code == 500
        assert "ReflexionTeachingAgent not initialized" in response.json()["detail"]

# A simple health check test
def test_health_check(client: TestClient):
    response = client.get("/health/")
    assert response.status_code == 200
    # The client fixture mocks the agent, so agent_initialized should be true if the mock is assigned
    # If the health check directly checks the class instance, this will reflect the mock.
    assert response.json() == {"status": "healthy", "agent_initialized": True}

def test_health_check_agent_really_none(client: TestClient):
    with patch('api.agent', new=None):
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "agent_initialized": False}

# Test for /process-lecture/ endpoint to ensure it also checks for agent initialization
def test_process_lecture_agent_not_initialized(client: TestClient):
    with patch('api.agent', new=None):
        # Prepare a dummy file for upload
        dummy_pdf_content = b"%PDF-1.4 fake pdf content"
        files = {'file': ('lecture.pdf', dummy_pdf_content, 'application/pdf')}

        response = client.post("/process-lecture/", files=files)
        assert response.status_code == 500
        assert "ReflexionTeachingAgent not initialized" in response.json()["detail"]

# Test for /analyze-code/ endpoint for agent initialization check
def test_analyze_code_agent_not_initialized(client: TestClient):
     with patch('api.agent', new=None):
        payload = {"code": "print('hello')", "exercise_id": "ex1"}
        response = client.post("/analyze-code/", json=payload)
        assert response.status_code == 500
        assert "ReflexionTeachingAgent not initialized" in response.json()["detail"]

# Note: The client fixture uses patch('api.ReflexionTeachingAgent') and patch('api.agent').
# This means that for most tests, api.agent will be a MagicMock instance.
# The tests for "agent_not_initialized" specifically re-patch api.agent to be None
# to test those specific error paths in the endpoints.
# The mock_env_vars_for_api fixture also mocks os.path.exists to prevent actual file operations
# during the setup of these API tests, which is good practice.

# The client fixture has been updated to patch 'api.agent' directly for better control.
# This ensures that when the TestClient is created, the global 'agent' in 'api.py'
# is the MagicMock instance we want for testing.

# Modification in client fixture to ensure 'api.agent' is the mock:
@pytest.fixture(scope="module")
def client_updated():
    """Provides a TestClient instance for API testing with direct agent patching."""
    # Create a mock instance for the agent
    mock_agent_instance = MagicMock(spec=ReflexionTeachingAgent)

    # Patch the 'agent' variable in the 'api' module
    with patch('api.agent', new=mock_agent_instance) as patched_agent:
        # The TestClient will now use the application with the patched 'agent'
        with TestClient(app) as test_client:
            # Make the mock agent accessible if needed, e.g., by attaching to app.state
            # FastAPI doesn't automatically put it in app.state unless you design it that way.
            # For these tests, we will retrieve it by re-importing or using the 'patched_agent' object
            # directly if the test needs to configure the mock.
            # A simple way is to ensure the mock object is the one from the patch context.
            test_client.app.state.agent = patched_agent # Storing for easier access if needed.
            yield test_client

# Re-run tests with this updated fixture logic if needed.
# For this script, I'll assume the original client fixture's patching strategy is generally okay,
# but direct patching of `api.agent` is more robust.
# The tests above are written assuming `api.agent` is successfully mocked by the fixture.
# The `test_grade_assessment_endpoint_success` was updated to reflect how to access the mock.
# The other tests for agent not initialized use a local patch to set api.agent to None.
